"""Full FP32-master/BF16-autocast encoder finetuning, DDP, restartable state.

No precomputed image/text features are used for gradient updates. The original
model is evaluated as epoch 0; a trained checkpoint must beat it on macro dev.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import __version__ as transformers_version

from ..common import fingerprint, read_jsonl, set_seed, sha256, write_json, write_jsonl
from .data import normalize_text, positive_map, validate_catalog
from .evaluate import all_images, evaluate_vectors, export_index
from .io import PairDataset, autocast, load_local, model_identity
from .model import FullSiglip, SiglipPairLoss, gather_indices, pair_masks


def rank0_value(fn):
    value = None
    error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            value = dict(ok=True,result=fn())
        except Exception as e:
            error = e
            value = dict(ok=False,error=f"{type(e).__name__}: {e}")
    if dist.is_initialized():
        values = [value]
        dist.broadcast_object_list(values,src=0)
        value = values[0]
    if not value["ok"]:
        raise RuntimeError(value["error"]) from error
    return value["result"]


def barrier():
    if dist.is_initialized():
        dist.barrier()


def lr_multiplier(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1,warmup)
    progress = (step-warmup) / max(1,total-warmup)
    return .5 * (1 + math.cos(math.pi * min(1.,progress)))


def save_state(path, model, optimizer, scheduler, progress, contract, device):
    """All ranks participate; only rank 0 writes. Previous last.pt stays valid on failure."""
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    if dist.is_initialized():
        states = [None]*dist.get_world_size()
        dist.all_gather_object(states,state)
    else:
        states = [state]
    def write():
        temporary = path.with_suffix(".tmp")
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), progress=progress, contract=contract, rng=states), temporary)
        temporary.replace(path)
    rank0_value(write)


def train(args):
    rank, world = int(os.environ.get("RANK",0)), int(os.environ.get("WORLD_SIZE",1))
    if args.device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK",0))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; keep the existing CUDA torch installation")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda",local_rank)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",timeout=timedelta(hours=2))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS",4)))
    config = json.loads(Path(args.config).read_text())
    if config["selection_metric"] != "macro_en_ru_image_ndcg5":
        raise ValueError("Unsupported selection metric")
    if not 0 <= config["warmup_fraction"] < 1 or min(config["epochs"],config["batch_size_per_gpu"],config["eval_batch_size"]) < 1:
        raise ValueError("Invalid training schedule")
    if config["precision"] not in {"bf16", "fp32"}:
        raise ValueError("Use bf16 or fp32")
    if device.type == "cuda" and config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unsupported on this GPU")
    set_seed(config["seed"])
    prepared, out = Path(args.prepared), Path(args.output)
    pairs = read_jsonl(prepared / "pairs.jsonl")
    items = read_jsonl(prepared / "items.jsonl")
    queries = read_jsonl(prepared / "queries.jsonl")
    by_id = validate_catalog(items,queries)
    for p in pairs:
        if p["item_id"] not in p["positive_item_ids"] or not p["weight"] > 0:
            raise ValueError("Invalid training pair")
        if any(by_id[pid]["split"] != "train" for pid in p["positive_item_ids"]):
            raise ValueError("Dev/test data leaked into training")
    if len(pairs) < 2 * world:
        raise ValueError("Too few training pairs")
    provenance = json.loads((prepared / "provenance.json").read_text())
    if sha256(Path(args.catalog)/"manifest.jsonl") != provenance["manifest_sha256"]:
        raise ValueError("Prepared catalog has changed")
    def make_contract():
        return dict(config=config, world_size=world, model_sha256=model_identity(args.model),
            input_hashes={name:sha256(prepared/name) for name in
                          ("pairs.jsonl", "items.jsonl", "queries.jsonl", "provenance.json")},
            code_hash=fingerprint({p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")}),
            torch_version=torch.__version__, transformers_version=transformers_version,
            smoke=args.smoke, device_type=device.type)
    contract = rank0_value(make_contract)
    def check_output():
        out.mkdir(parents=True,exist_ok=True)
        if (out / "config.json").exists():
            if not args.resume:
                raise ValueError("Run already exists; use --resume or a new --output")
            if json.loads((out / "config.json").read_text()) != contract:
                raise ValueError("Resume inputs/config/code/world size differ; use a new run")
        write_json(out / "config.json",contract)
        return (out / "complete.json").exists()
    if rank0_value(check_output):
        if rank == 0:
            print(f"Already complete: {out}",flush=True)
        return
    backbone, processor = load_local(args.model,device)
    if config["max_text_length"] > backbone.config.text_config.max_position_embeddings:
        raise ValueError("Text length exceeds the model positional embeddings")
    if config["gradient_checkpointing"]:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model = FullSiglip(backbone)
    dataset = PairDataset(pairs,items,args.catalog,processor,config["max_text_length"])
    positives = positive_map(pairs,processor.tokenizer,config["max_text_length"])
    item_groups = {i["item_id"]:i["group_id"] for i in items}
    sampler = DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=config["seed"],drop_last=False)
    workers = config["num_workers"]
    loader = DataLoader(dataset,batch_size=config["batch_size_per_gpu"],sampler=sampler,
        num_workers=workers,pin_memory=device.type == "cuda",drop_last=False,
        generator=torch.Generator().manual_seed(config["seed"]+rank),
        **({"multiprocessing_context":"spawn", "persistent_workers":True} if workers else {}))
    if len(loader) < 1:
        raise ValueError("Empty loader")
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or "embedding" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params":decay,"weight_decay":config["weight_decay"]},
        {"params":no_decay,"weight_decay":0.0}],lr=config["learning_rate"])
    total_steps = len(loader)*config["epochs"]
    warmup = max(1,round(total_steps*config["warmup_fraction"])) if config["warmup_fraction"] else 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_multiplier(step,warmup,total_steps))
    progress = dict(epoch=0,next_batch=0,global_step=0,history=[],selection=None)
    last = out / "last.pt"
    if args.resume and last.exists():
        # Only load our own local training state (contains Python/NumPy RNG state).
        saved = torch.load(last,map_location="cpu",weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("Checkpoint contract mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        progress = saved["progress"]
        rng = saved["rng"][rank]
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"],device)
        del saved
    if world > 1:
        network = DistributedDataParallel(model,device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,gradient_as_bucket_view=True)
    else:
        network = model
    loss_fn = SiglipPairLoss()
    if rank == 0:
        stats = dict(total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            train_unique_images=len({p["item_id"] for p in pairs}), training_pairs=len(pairs),
            global_batch=config["batch_size_per_gpu"]*world, steps_per_epoch=len(loader),
            distributed_sampler_padding=len(sampler)*world-len(dataset), loss="weighted multi-positive sigmoid",
            negative_scope="current global batch, differentiable image all_gather",
            optimizer="AdamW", precision="FP32 master weights; BF16 autocast" if device.type == "cuda" else "FP32 CPU")
        lengths = processor.tokenizer([normalize_text(p["text"]) for p in pairs],truncation=False,padding=False)["input_ids"]
        stats["texts_truncated"] = sum(len(t)>config["max_text_length"] for t in lengths)
        write_json(out / "training_data.json",stats)
        write_jsonl(out / "queries.jsonl",queries)
        print(json.dumps(stats,ensure_ascii=False),flush=True)
    barrier()

    def assess(epoch):
        vectors = all_images(backbone,processor,items,args.catalog,device,config)
        def compute():
            result = evaluate_vectors(backbone,processor,vectors,items,queries,device,config,"dev",
                out / "evaluations" / f"epoch_{epoch:03d}")
            print(json.dumps({"epoch":epoch,"dev_macro_ndcg5":result["macro_en_ru_image_ndcg5"],
                **{f"dev_{lang}_ndcg5":m["NDCG@5"]["mean"] for lang,m in result["metrics"].items()}}),flush=True)
            return result
        result = rank0_value(compute)
        return vectors,result

    if not args.smoke and progress["selection"] is None:
        vectors,result = assess(0)
        rank0_value(lambda:export_index(out/"baseline",vectors,items,args.model,config,provenance["manifest_sha256"]))
        progress["selection"] = dict(epoch=0,metric=result["macro_en_ru_image_ndcg5"],
            model_path=str(Path(args.model).resolve()), index_path=str((out/"baseline").resolve()),
            selection_split="dev", selection_metric=config["selection_metric"])
        progress["history"] = [dict(epoch=0,loss=None,dev=result)]
        save_state(last,model,optimizer,scheduler,progress,contract,device)
        del vectors
    if rank == 0 and progress["selection"] is not None:
        write_json(out/"selection.json",progress["selection"])
        write_jsonl(out/"history.jsonl",progress["history"])
    probe_names = ("vision_model.embeddings.patch_embedding.weight", "text_model.head.weight")
    probes = {name:dict(backbone.named_parameters())[name] for name in probe_names}
    before = {name:p.detach().flatten()[:128].clone() for name,p in probes.items()}
    smoke_grads = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    start_step = progress["global_step"]
    for epoch in range(progress["epoch"],config["epochs"]):
        sampler.set_epoch(epoch)
        network.train()
        loss_sum = 0.0
        steps = 0
        begin = progress["next_batch"] if epoch == progress["epoch"] else 0
        for batch_number,batch in enumerate(loader):
            if batch_number < begin:
                continue
            indices = batch.pop("row_index").to(device)
            weights = batch.pop("weight").to(device,dtype=torch.float32)
            inputs = {k:v.to(device,non_blocking=True) for k,v in batch.items()}
            global_indices = gather_indices(indices)
            pos,valid,column_weights = pair_masks(indices.tolist(),global_indices,pairs,positives,item_groups,device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device,config["precision"]):
                images,texts,scale,bias = network(**inputs)
            loss = loss_fn(images,texts,scale,bias,pos,valid,weights,column_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config["max_grad_norm"],error_if_nonfinite=True)
            if args.smoke:
                smoke_grads = {name:float(p.grad.float().norm()) if p.grad is not None else 0 for name,p in probes.items()}
            optimizer.step()
            scheduler.step()
            loss_value = loss.detach().clone()
            if world > 1:
                dist.all_reduce(loss_value)
                loss_value /= world
            loss_sum += float(loss_value); steps += 1
            progress.update(epoch=epoch,next_batch=batch_number+1,global_step=progress["global_step"]+1)
            if rank == 0 and (steps == 1 or progress["global_step"] % config["log_every_steps"] == 0):
                print(json.dumps(dict(epoch=epoch+1,step=progress["global_step"],steps_total=total_steps,
                    loss=float(loss_value),lr=optimizer.param_groups[0]["lr"],grad_norm=float(grad_norm),
                    seconds_per_step=(time.perf_counter()-start_time)/(progress["global_step"]-start_step))),flush=True)
            if args.smoke and steps >= 2:
                updates = {name:float((p.detach().flatten()[:128]-before[name]).abs().max()) for name,p in probes.items()}
                if not all(v > 0 for v in smoke_grads.values()) or not all(v > 0 for v in updates.values()):
                    raise RuntimeError("Smoke check failed: one of the encoder towers did not update")
                reports = [None]*world
                report = dict(rank=rank,grad_norms=smoke_grads,parameter_max_changes=updates,
                    peak_cuda_allocated_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type=="cuda" else None,
                    peak_cuda_reserved_gb=torch.cuda.max_memory_reserved(device)/1e9 if device.type=="cuda" else None)
                if world > 1:
                    dist.all_gather_object(reports,report)
                else:
                    reports = [report]
                if rank == 0:
                    write_json(out/"complete.json",dict(smoke=True,steps=2,ranks=reports))
                    print("SMOKE PASSED: both encoder towers changed; all-GPU gradients are finite",flush=True)
                return
            if not args.smoke and config["save_every_steps"] and progress["global_step"] % config["save_every_steps"] == 0:
                save_state(last,model,optimizer,scheduler,progress,contract,device)
            if not args.smoke and args.stop_after_steps and progress["global_step"] >= args.stop_after_steps:
                save_state(last,model,optimizer,scheduler,progress,contract,device)
                if rank == 0:
                    print(f"Paused after step {progress['global_step']}; restart with --resume",flush=True)
                return
        if args.smoke:
            raise ValueError("Smoke requires at least two batches")
        vectors,result = assess(epoch+1)
        score = result["macro_en_ru_image_ndcg5"]
        if score > progress["selection"]["metric"]:
            checkpoint = out / "checkpoints" / f"epoch_{epoch+1:03d}"
            def save_best():
                checkpoint.mkdir(parents=True,exist_ok=True)
                backbone.save_pretrained(checkpoint/"model",safe_serialization=True)
                processor.save_pretrained(checkpoint/"model")
                export_index(checkpoint/"index",vectors,items,checkpoint/"model",config,provenance["manifest_sha256"])
            rank0_value(save_best)
            progress["selection"] = dict(epoch=epoch+1,metric=score,
                model_path=str((checkpoint/"model").resolve()),index_path=str((checkpoint/"index").resolve()),
                selection_split="dev",selection_metric=config["selection_metric"])
        del vectors
        progress["history"].append(dict(epoch=epoch+1,loss=loss_sum/steps if steps else None,
            loss_batches_in_this_process=steps,resumed_mid_epoch=begin>0,dev=result))
        progress.update(epoch=epoch+1,next_batch=0)
        save_state(last,model,optimizer,scheduler,progress,contract,device)
        if rank == 0:
            write_json(out/"selection.json",progress["selection"])
            write_jsonl(out/"history.jsonl",progress["history"])
    if rank == 0:
        write_json(out/"complete.json",dict(selection=progress["selection"],global_step=progress["global_step"],
            epochs=config["epochs"],test_evaluated=False,finetuning_improved_dev=progress["selection"]["epoch"]>0))
    barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",default="configs/siglip2_full.json")
    p.add_argument("--model",required=True,help="Offline local HF snapshot")
    p.add_argument("--catalog",default="data/catalog")
    p.add_argument("--prepared",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--stop-after-steps",type=int,default=0,help="Optional controlled pause for restart verification")
    args = p.parse_args()
    try:
        train(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
