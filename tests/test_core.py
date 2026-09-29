import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sticker_search.annotate import parse_answer
from sticker_search.common import image_properties, sha256, write_json, write_jsonl
from sticker_search.metrics import query_metrics
from sticker_search.model import multi_positive_loss
from sticker_search.pack import pack, unpack
from sticker_search.prepare import group_and_split


def test_metrics_multiple_positives_and_grades():
    result=query_metrics(['x','b','a'],{'a':2,'b':1},ks=(1,3))
    assert result['Hit@1']==0
    assert result['Recall@3']==1
    assert result['MRR@10']==.5
    expected=(1/np.log2(3)+3/np.log2(4))/(3+1/np.log2(3))
    assert result['NDCG@3']==pytest.approx(expected)


def test_contrastive_masks_and_gradients():
    scores=torch.tensor([[2.,1.,100.],[0.,3.,-1.]],requires_grad=True)
    positives=torch.tensor([[True,True,False],[False,True,False]])
    valid=torch.tensor([[True,True,False],[True,True,True]])
    loss=multi_positive_loss(scores,positives,valid,1.)
    assert loss.item()==pytest.approx(float(torch.logsumexp(scores.detach()[1],0)-3)/2)
    loss.backward()
    assert scores.grad[0,2]==0
    assert scores.grad[1,1]<0
    assert torch.isfinite(scores.grad).all()


def test_hidden_rgb_is_not_a_new_image(tmp_path):
    a=Image.new('RGBA',(10,10),(255,0,0,0));b=Image.new('RGBA',(10,10),(0,255,0,0))
    a.putpixel((5,5),(1,2,3,255));b.putpixel((5,5),(1,2,3,255))
    a.save(tmp_path/'a.png');b.save(tmp_path/'b.png')
    assert image_properties(tmp_path/'a.png')['pixel_sha256']==image_properties(tmp_path/'b.png')['pixel_sha256']


def test_known_family_never_crosses_split():
    rows=[dict(item_id=f'i{i}',source_family='family',dhash=f'{i:016x}',kind='sticker',width=10,height=10,mean_rgb=[100]*3) for i in range(4)]
    group_and_split(rows)
    assert len({r['split'] for r in rows})==1
    assert len({r['group_id'] for r in rows})==1


def test_shared_layout_does_not_merge_known_different_families():
    rows=[dict(item_id=f'i{i}',source_family=f'artwork{i}',dhash='0'*16,kind='postcard',width=10,height=10,mean_rgb=[100]*3) for i in range(2)]
    group_and_split(rows)
    assert len({r['group_id'] for r in rows})==2


def test_foreground_guard_rejects_dhash_collision(tmp_path):
    from PIL import ImageDraw
    rows=[]
    for i,color in enumerate(['red','blue']):
        im=Image.new('RGB',(100,100),'white');ImageDraw.Draw(im).rectangle((30,30,70,70),fill=color)
        im.save(tmp_path/f'{i}.png')
        rows.append(dict(item_id=str(i),source_family=None,dhash='0'*16,kind='sticker',
            width=100,height=100,mean_rgb=[250]*3,image_path=f'{i}.png'))
    group_and_split(rows,image_root=tmp_path)
    assert len({r['group_id'] for r in rows})==2


def test_annotation_must_validate_schema():
    assert parse_answer('```json\n{"text_ru":"привет"}\n```','translate')=={'text_ru':'привет'}
    with pytest.raises(ValueError):
        parse_answer('{"text_ru":""}','translate')
    with pytest.raises(ValueError):
        parse_answer('{"caption_ru":"кошка"}','caption')


def test_bundle_roundtrip(tmp_path):
    root=tmp_path/'source';(root/'images').mkdir(parents=True)
    Image.new('RGB',(8,8),'red').save(root/'images/a.png')
    row=dict(item_id='a',image_path='images/a.png',file_sha256=sha256(root/'images/a.png'))
    write_jsonl(root/'manifest.jsonl',[row])
    pack(root,tmp_path/'bundle',1)
    unpack(tmp_path/'bundle',tmp_path/'restored')
    assert (root/'images/a.png').read_bytes()==(tmp_path/'restored/images/a.png').read_bytes()


def test_bundle_rejects_traversal(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    bad=bundle/'bad.tar'
    with tarfile.open(bad,'w') as tar:
        item=tarfile.TarInfo('../escaped');item.size=1;tar.addfile(item,io.BytesIO(b'x'))
    write_json(bundle/'bundle.json',dict(num_images=0,files=[dict(file='bad.tar',sha256=sha256(bad))]))
    with pytest.raises(ValueError,match='Unsafe'):
        unpack(bundle,tmp_path/'out')
    assert not (tmp_path/'escaped').exists()
