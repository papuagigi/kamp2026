"""Package only the agreed dataset, Faster weights, and portable execution files."""
import hashlib
import json
from pathlib import Path
import zipfile
from xray_config import ROOT

def main():
    report=ROOT/'reports/overnight_20261003';report.mkdir(parents=True,exist_ok=True)
    dest=report/'xray_colab_t4.zip'
    files=[]
    for p in (ROOT/'data/xray_combined_20261003').rglob('*'):
        if p.is_file() and p.suffix in {'.png','.txt','.json','.yaml'}:files.append(p)
    for split in ['val','test']:
        for kind in ['images','labels']:files.extend(p for p in (ROOT/'data/xray_v2'/kind/split).iterdir() if p.suffix in {'.png','.txt'})
    files.extend(ROOT/p for p in ['data/xray_v2/manifest.csv','data/xray_v2/split.csv',
        'weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',
        'scripts/xray_config.py','scripts/xray_model_io.py','scripts/xray_train_resumable.py','scripts/xray_recovery.py',
        'scripts/xray_predict_controlled.py','scripts/xray_eval.py','scripts/xray_evaluate_run.py',
        'scripts/xray_colab_run.py','reports/overnight_20261003/evaluation_policy.json',
        'reports/overnight_20261003/data_audit.json'])
    checksums={}
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(set(files)):
            arc=str(p.relative_to(ROOT));checksums[arc]=hashlib.sha256(p.read_bytes()).hexdigest();z.write(p,arc)
        z.writestr('bundle_checksums.json',json.dumps(checksums,indent=2))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    result=dict(path=str(dest.relative_to(ROOT)),files=len(checksums),bytes=dest.stat().st_size,
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),original_raw_included=False,contains_only_faster_weights=True)
    (report/'colab_package.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
