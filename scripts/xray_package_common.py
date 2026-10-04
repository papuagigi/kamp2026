"""Create the new train/validation-only CUDA bundle; preserve the previous bundle."""
import json
import zipfile
import importlib.metadata
from xray_config import ROOT
from xray_recovery import sha256, atomic_json


def main():
    out=ROOT/'reports/common_epoch_20261004';dest=out/'xray_common_t4_20261004.zip'
    sources=set()
    for folder in ['data/xray_combined_20261003','reports/recovery_preparation_20261004/smoke_data']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and 'test' not in p.relative_to(ROOT/folder).parts and p.suffix in {'.png','.txt','.json','.yaml'}:sources.add(p)
    for kind in ['images','labels']:
        sources.update((ROOT/'data/xray_v2'/kind/'val').iterdir())
    sources.update(ROOT/'data/xray_v2'/n for n in ['manifest.csv','split.csv'])
    sources.update(ROOT/'weights'/n for n in ['fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth','dfine_s_coco.pth'])
    sources.update(ROOT/'scripts'/n for n in ['xray_config.py','xray_model_io.py','xray_recovery.py',
        'xray_train_resumable.py','xray_predict_controlled.py','xray_eval.py','xray_epoch_validation.py',
        'xray_dfine.py','xray_common_queue.py','test_xray_recovery.py','test_xray_epoch_validation.py'])
    vendor=out/'vendor/D-FINE'
    sources.update(p for p in vendor.rglob('*') if p.is_file() and '.git' not in p.parts and
                   (p.suffix in {'.py','.yml','.yaml'} or p.name in {'LICENSE','README.md','requirements.txt'}))
    sources.add(out/'evaluation_policy.json')
    deps={n:importlib.metadata.version(n) for n in ['numpy','pandas','Pillow','PyYAML','opencv-python-headless',
        'scipy','faster-coco-eval','transformers','calflops','loguru','tensorboard']}
    lock=out/'colab_requirements.txt';lock.write_text('\n'.join(k+'=='+v for k,v in deps.items())+'\n');sources.add(lock)
    checksums={}
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(sources):
            arc=str(p.relative_to(ROOT));checksums[arc]=sha256(p);z.write(p,arc)
        z.writestr('bundle_checksums.json',json.dumps(checksums,indent=2))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    result=dict(path=str(dest.relative_to(ROOT)),sha256=sha256(dest),files=len(checksums),bytes=dest.stat().st_size,
        test_images_included=False,source_commit='956d1709314c2c6a4df6f34de232054578a7449f')
    atomic_json(out/'package.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
