"""Actual catalog EDA: counts, dimensions, transparency, groups and contact sheet."""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

from .common import read_jsonl,rgb_image,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',default='data/catalog');p.add_argument('--output',default='evidence/catalog')
    args=p.parse_args();root=Path(args.catalog);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(root/'manifest.jsonl')
    source=Counter(r['source'] for r in rows);kind=Counter(r['kind'] for r in rows)
    groups=Counter(r['group_id'] for r in rows)
    summary=dict(images=len(rows),sources=dict(source),kinds=dict(kind),splits=dict(Counter(r['split'] for r in rows)),
        transparent_images=sum(r['has_transparency'] for r in rows),unique_pixel_hashes=len({r['pixel_sha256'] for r in rows}),
        width_quantiles=np.quantile([r['width'] for r in rows],[0,.5,.95,1]).tolist(),
        height_quantiles=np.quantile([r['height'] for r in rows],[0,.5,.95,1]).tolist(),
        num_groups=len(groups),largest_group=max(groups.values()),
        source_split_counts={s:dict(Counter(r['split'] for r in rows if r['source']==s)) for s in source},
        limitation='No semantic quality/translation/OCR accuracy measured in this catalog-only audit.')
    write_json(out/'summary.json',summary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axes[0,0].barh(list(source),list(source.values()),color='#536dfe');axes[0,0].set_title('Images by source')
    for i,v in enumerate(source.values()):axes[0,0].text(v,i,f' {v:,}',va='center')
    splits=Counter(r['split'] for r in rows);axes[0,1].bar(list(splits),list(splits.values()),color='#18a999');axes[0,1].set_title('Grouped image split')
    axes[1,0].hist([r['alpha_fraction'] for r in rows],bins=20,color='#536dfe');axes[1,0].set_title('Fraction of non-opaque pixels');axes[1,0].set_xlabel('alpha < 255')
    bins=[('1',1,1),('2–5',2,5),('6–10',6,10),('11–50',11,50),('51+',51,10**9)]
    counts=[sum(lo<=v<=hi for v in groups.values()) for _,lo,hi in bins]
    axes[1,1].bar([name for name,_,_ in bins],counts,color='#e6a23c');axes[1,1].set_yscale('log');axes[1,1].set_title('Duplicate/family group sizes');axes[1,1].set_xlabel('Images in group')
    fig.savefig(out/'eda.png',dpi=150);plt.close(fig)
    rng=np.random.default_rng(42);selected=[]
    for s in source:
        candidates=[r for r in rows if r['source']==s]
        selected.extend(candidates[i] for i in rng.choice(len(candidates),size=min(12,len(candidates)),replace=False))
    w,h=180,205;cols=6;canvas=Image.new('RGB',(cols*w,((len(selected)+cols-1)//cols)*h),'#f4f5fa');draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(selected):
        x,y=(i%cols)*w,(i//cols)*h
        im=ImageOps.contain(rgb_image(root/r['image_path']),(160,165))
        canvas.paste(im,(x+(w-im.width)//2,y+5+(165-im.height)//2))
        draw.text((x+7,y+173),r['source'][:24],fill='#25283a')
        draw.text((x+7,y+188),r['item_id'][:22],fill='#6a6f80')
    canvas.save(out/'contact_sheet.jpg',quality=90)
    print(summary)


if __name__=='__main__':main()
