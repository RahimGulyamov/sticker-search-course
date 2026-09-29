"""Meaningful CPU checks for split protection and exact distributed gradients."""
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from sticker_search.siglip.data import positive_map, text_problem, validate_catalog
from sticker_search.siglip.model import SiglipPairLoss, pair_masks


def test_split_and_duplicate_protection():
    items = [dict(item_id="a",group_id="g",split="train"),dict(item_id="b",group_id="g",split="test")]
    with pytest.raises(ValueError,match="crosses"):
        validate_catalog(items,[])
    items[1]["group_id"] = "other"
    q = dict(query_id="q",text="cat",positive_item_ids=["b"],split="train")
    with pytest.raises(ValueError,match="Cross-split"):
        validate_catalog(items,[q])
    pairs = [dict(text="CAT",item_id="a",positive_item_ids=["a"]),
             dict(text=" cat ",item_id="b",positive_item_ids=["b"]),
             dict(text="bird",item_id="c",positive_item_ids=["c"])]
    positives = positive_map(pairs)
    assert positives[:2] == [{"a","b"},{"a","b"}]
    pos,valid,w = pair_masks([0],[0,1,0,2],pairs,positives,{"a":"x","b":"y","c":"x"},"cpu")
    assert pos.tolist() == [[True,True,True,False]]
    assert valid.tolist() == [[True,True,True,False]]
    assert w.tolist() == [.5,1,.5,1]
    assert text_problem("Н"+"у"*100) == "long_or_repetitive"


def test_duplicate_columns_do_not_change_loss():
    fn = SiglipPairLoss()
    image = torch.tensor([[1.,0.],[0.,1.]])
    text = torch.tensor([[.8,.2]])
    first = fn(image,text,torch.tensor(2.),torch.tensor(-1.),torch.tensor([[True,False]]),
               torch.ones(1,2,dtype=torch.bool),torch.ones(1),torch.ones(2))
    second = fn(image[[0,0,1]],text,torch.tensor(2.),torch.tensor(-1.),torch.tensor([[True,True,False]]),
                torch.ones(1,3,dtype=torch.bool),torch.ones(1),torch.tensor([.5,.5,1.]))
    torch.testing.assert_close(first,second)


def test_prepare_uses_only_train_captions_and_reports_missing(tmp_path):
    import json
    from PIL import Image
    from sticker_search.common import read_jsonl, write_jsonl
    from sticker_search.siglip.data import prepare
    catalog=tmp_path/"catalog"
    catalog.mkdir()
    items=[]
    annotations=[]
    for i,split in enumerate(("train","train","dev","test")):
        pid=f"p{i}"
        Image.new("RGB",(8,8),(i*50,10,20)).save(catalog/f"{pid}.png")
        items.append(dict(item_id=pid,group_id=pid,split=split,image_path=f"{pid}.png"))
        if i != 1:
            annotations.append(dict(item_id=pid,caption_en="a happy cat",caption_ru="веселый кот"))
    write_jsonl(catalog/"manifest.jsonl",items)
    write_jsonl(catalog/"labels/queries_en.jsonl",[])
    write_jsonl(tmp_path/"ru.jsonl",[])
    write_jsonl(tmp_path/"ann.jsonl",annotations)
    config=dict(min_train_coverage=.5,caption_weight=.5,source_query_weight=1.,translated_query_weight=.7)
    prepare(catalog,tmp_path/"ann.jsonl",tmp_path/"ru.jsonl",tmp_path/"out",config)
    pairs=read_jsonl(tmp_path/"out/pairs.jsonl")
    assert len(pairs)==2 and {p["item_id"] for p in pairs}=={"p0"}
    coverage=json.loads((tmp_path/"out/coverage.json").read_text())
    assert coverage["missing_train_item_ids"]==["p1"]
    config["min_train_coverage"]=.98
    with pytest.raises(ValueError,match="Insufficient"):
        prepare(catalog,tmp_path/"ann.jsonl",tmp_path/"ru.jsonl",tmp_path/"blocked",config)


def test_ocr_fusion_has_no_fake_matches():
    import numpy as np
    from sticker_search.siglip.search import OcrFusion
    raw=np.array([.9,.8,.1],dtype=np.float32)
    fusion=OcrFusion([dict(ocr_text=""),dict(ocr_text="happy birthday"),dict(ocr_text="good night")])
    np.testing.assert_equal(fusion.scores("кот",raw),raw)
    assert fusion.scores("happy birthday",raw).argmax()==1


class TinyDual(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(3,2,bias=False)
        self.text = nn.Linear(3,2,bias=False)
        self.scale = nn.Parameter(torch.tensor(1.2))
        self.bias = nn.Parameter(torch.tensor(-.3))

    def forward(self, x,y):
        return self.image(x),self.text(y),self.scale,self.bias


def _ddp_worker(rank, initfile, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",rank=rank,world_size=2,init_method="file://"+initfile)
    torch.manual_seed(123)
    model = TinyDual()
    ddp = DDP(model)
    x = torch.arange(12,dtype=torch.float32).reshape(4,3)/12
    y = torch.flip(x,[0])
    local = slice(rank*2,rank*2+2)
    images,texts,scale,bias = ddp(x[local],y[local])
    weights = torch.tensor([.5,1.,.7,.5])
    positives = torch.eye(4,dtype=torch.bool)
    positives[0,1] = True  # Multi-positive and one masked unknown pair.
    valid = torch.ones(4,4,dtype=torch.bool)
    valid[2,3] = False
    loss = SiglipPairLoss()(images,texts,scale,bias,positives[local],valid[local],weights[local],torch.ones(4))
    loss.backward()
    if rank == 0:
        torch.save({n:p.grad for n,p in model.named_parameters()},output)
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("STICKER_TEST_DDP") != "1",reason="Set STICKER_TEST_DDP=1 where Gloo sockets are permitted")
def test_ddp_gradients_match_global_batch(tmp_path):
    output = str(tmp_path/"grad.pt")
    mp.spawn(_ddp_worker,args=(str(tmp_path/"init"),output),nprocs=2,join=True)
    torch.manual_seed(123)
    model = TinyDual()
    x = torch.arange(12,dtype=torch.float32).reshape(4,3)/12
    images,texts,scale,bias = model(x,torch.flip(x,[0]))
    positives = torch.eye(4,dtype=torch.bool); positives[0,1] = True
    valid = torch.ones(4,4,dtype=torch.bool); valid[2,3] = False
    loss = SiglipPairLoss()(images,texts,scale,bias,positives,valid,torch.tensor([.5,1.,.7,.5]),torch.ones(4))
    loss.backward()
    distributed = torch.load(output,weights_only=True)
    for name,p in model.named_parameters():
        torch.testing.assert_close(p.grad,distributed[name],rtol=1e-5,atol=1e-6)


def test_sharded_weight_normalization_matches_global(monkeypatch):
    """Check distributed loss algebra without claiming to test network transport."""
    import sticker_search.siglip.model as module
    torch.manual_seed(91)
    image = torch.randn(4,3,requires_grad=True)
    text = torch.randn(4,3,requires_grad=True)
    scale = torch.tensor(1.2,requires_grad=True)
    bias = torch.tensor(-.5,requires_grad=True)
    weights = torch.tensor([.5,1.,.7,.5])
    positive = torch.eye(4,dtype=torch.bool)
    valid = torch.ones_like(positive)
    fn = SiglipPairLoss()
    reference = fn(image,text,scale,bias,positive,valid,weights,torch.ones(4))
    expected = torch.autograd.grad(reference,(image,text,scale,bias),retain_graph=True)
    monkeypatch.setattr(module.dist,"is_initialized",lambda:True)
    monkeypatch.setattr(module.dist,"get_world_size",lambda:2)
    monkeypatch.setattr(module.dist,"all_reduce",lambda tensor:tensor.copy_(weights.sum()))
    monkeypatch.setattr(module,"all_gather",lambda tensor:(image[:2],image[2:]))
    shards = [fn(image[s:s+2],text[s:s+2],scale,bias,positive[s:s+2],valid[s:s+2],weights[s:s+2],torch.ones(4)) for s in (0,2)]
    actual = torch.autograd.grad(sum(shards)/2,(image,text,scale,bias))
    for left,right in zip(expected,actual):
        torch.testing.assert_close(left,right)
