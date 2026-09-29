"""Create a tiny random SiglipModel and run the real CPU pipeline offline.

These results verify software behavior; they are NOT project retrieval metrics.
Run: python tests/siglip2_integration.py /tmp/siglip2-integration
An optional second argument 2 enables actual DDP where sockets are available.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GemmaTokenizerFast, SiglipConfig, SiglipImageProcessor, SiglipModel, SiglipProcessor

from sticker_search.common import write_json, write_jsonl
from sticker_search.siglip.data import prepare


def main(root,workers=1):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Use a fresh integration-test directory")
    root.mkdir(parents=True)
    torch.manual_seed(17)
    words = ["<pad>","<eos>","<bos>","<unk>","red","blue","cat","dog","happy","sad",
             "кот","собака","веселый","грустный","sticker","with","a","стикер"]
    raw = Tokenizer(WordLevel({word:i for i,word in enumerate(words)},unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = GemmaTokenizerFast(tokenizer_object=raw,pad_token="<pad>",eos_token="<eos>",
        bos_token="<bos>",unk_token="<unk>",add_bos_token=False,add_eos_token=True,model_max_length=16)
    proc = SiglipProcessor(image_processor=SiglipImageProcessor(size={"height":16,"width":16}),tokenizer=tok)
    text = dict(vocab_size=len(tok),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                num_attention_heads=2,max_position_embeddings=16,projection_size=16)
    vision = dict(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
                  image_size=16,patch_size=4)
    model = SiglipModel(SiglipConfig(text_config=text,vision_config=vision))
    model.save_pretrained(root/"model")
    proc.save_pretrained(root/"model")
    catalog = root/"catalog"
    (catalog/"images").mkdir(parents=True)
    items,annotations,en,ru = [],[],[],[]
    for i in range(10):
        split = "train" if i < 6 else "dev" if i < 8 else "test"
        pid = f"item_{i}"
        Image.new("RGB",(20,20),((i*27)%255,100,(255-i*17)%255)).save(catalog/"images"/f"{pid}.png")
        item = dict(item_id=pid,group_id=pid,split=split,image_path=f"images/{pid}.png",kind="sticker",source="SOFTWARE_TEST_ONLY")
        items.append(item)
        annotations.append(dict(item_id=pid,caption_en="happy cat sticker" if i%2 else "sad dog sticker",
                                caption_ru="веселый кот стикер" if i%2 else "грустный собака стикер",ocr_text=""))
        q = dict(query_id=f"q{i}_en",text="happy cat" if i%2 else "sad dog",positive_item_ids=[pid],
                 group_id=pid,split=split,language="en",label_source="human_source")
        en.append(q)
        ru.append(q|dict(query_id=f"q{i}_ru",text="веселый кот" if i%2 else "грустный собака",
                         language="ru",label_source="human_source_machine_translation"))
    write_jsonl(catalog/"manifest.jsonl",items)
    write_jsonl(catalog/"labels/queries_en.jsonl",en)
    write_jsonl(root/"translations.jsonl",ru)
    write_jsonl(root/"annotations.jsonl",annotations)
    config = json.loads(Path("configs/siglip2_full.json").read_text())
    config.update(epochs=2,batch_size_per_gpu=4//workers,eval_batch_size=3,num_workers=0,precision="fp32",
                  max_text_length=16,learning_rate=1e-3,save_every_steps=3,log_every_steps=3)
    write_json(root/"config.json",config)
    prepare(catalog,root/"annotations.jsonl",root/"translations.jsonl",root/"prepared",config)
    launcher = [sys.executable] if workers == 1 else [sys.executable,"-m","torch.distributed.run","--standalone",f"--nproc_per_node={workers}"]
    base = launcher + ["-m","sticker_search.siglip.train","--device","cpu","--model",str(root/"model"),
            "--config",str(root/"config.json"),"--catalog",str(catalog),"--prepared",str(root/"prepared")]
    env = os.environ | {"OMP_NUM_THREADS":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                       "TOKENIZERS_PARALLELISM":"false"}
    for name,extras in [("smoke",["--smoke"]),("continuous",[]),
                        ("resumed",["--stop-after-steps","3"]),("resumed",["--resume"])]:
        subprocess.run(base+["--output",str(root/name)]+extras,env=env,check=True)
    a = torch.load(root/"continuous/last.pt",map_location="cpu",weights_only=False)
    b = torch.load(root/"resumed/last.pt",map_location="cpu",weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key],b["model"][key],atol=1e-7,rtol=1e-6)
    assert a["progress"]["global_step"] == b["progress"]["global_step"] == 12
    for which in ("baseline","selected"):
        subprocess.run([sys.executable,"-m","sticker_search.siglip.evaluate","--run",str(root/"continuous"),
            "--catalog",str(catalog),"--device","cpu","--split","dev","--which",which],env=env,check=True)
    from sticker_search.siglip.io import load_local, encode_texts
    from sticker_search.siglip.model import FullSiglip
    trained,processor=load_local(root/"model","cpu")
    FullSiglip(trained).load_state_dict(a["model"])
    expected=encode_texts(trained,processor,["happy cat","веселый кот"],"cpu",2,"fp32",16)
    trained.save_pretrained(root/"trained_export")
    processor.save_pretrained(root/"trained_export")
    restored,processor=load_local(root/"trained_export","cpu")
    actual=encode_texts(restored,processor,["happy cat","веселый кот"],"cpu",2,"fp32",16)
    torch.testing.assert_close(torch.from_numpy(expected),torch.from_numpy(actual))
    summary = dict(software_test_only=True,processes=workers,steps=12,full_resume_matches=True,
        both_towers_updated=True,baseline_and_selected_index_reload=True,trained_hf_export_reload=True,
        dataloader_workers=0,offline=True,
        torch=torch.__version__,fixture="Tiny randomly initialized SiglipModel; NOT pretrained SigLIP 2")
    write_json(root/"verification.json",summary)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 1)
