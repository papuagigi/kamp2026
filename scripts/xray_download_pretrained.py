"""Fetch exact official pretrained weights, verify SHA-256, preserve existing files."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
from xray_config import ROOT

SPECS={
'yolo':('yolov8n.pt','https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt','f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'),
'faster':('fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth','https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth','dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf'),
'rfdetr':('rf-detr-small.pth','https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth','d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33')}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['all',*SPECS],default='all');a=p.parse_args()
    (ROOT/'weights').mkdir(exist_ok=True)
    for kind,(name,url,sha) in SPECS.items():
        if a.model not in ['all',kind]:continue
        path=ROOT/'weights'/name
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.download')
            if temporary.exists():raise FileExistsError(f'Incomplete download preserved: {temporary}')
            with urllib.request.urlopen(url,timeout=120) as source,temporary.open('xb') as target:
                while block:=source.read(1024*1024):target.write(block)
            if digest(temporary)!=sha:raise ValueError(f'Weight hash mismatch; file preserved: {temporary}')
            temporary.replace(path)
        if digest(path)!=sha:raise ValueError(f'Existing weight differs; not overwritten: {path}')
        print(kind,'SHA-256 verified',flush=True)

if __name__=='__main__':main()
