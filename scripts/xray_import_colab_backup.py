"""Copy downloaded Drive artifacts without overwriting differing local results."""
import argparse
import json
import shutil
from pathlib import Path
from xray_config import ROOT
from xray_recovery import sha256, atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('sources',nargs='+',type=Path);a=p.parse_args()
    inventory=json.loads((ROOT/'reports/common_epoch_20261004/drive_backup_inventory.json').read_text())['files']
    mapped={}
    for source in a.sources:
        for path in source.rglob('*'):
            if not path.is_file() or path.is_symlink():continue
            parts=path.parts
            for i,name in enumerate(parts):
                if name in ['v2_common20_faster_cuda_20261004','v2_common20_dfine_cuda_20261004']:
                    key='runs/'+str(Path(*parts[i:]));mapped.setdefault(key,[]).append(path);break
    rows=[];missing=[]
    for item in inventory:
        dest=ROOT/item['path'];expected=int(item['size']);candidates=mapped.get(item['path'],[])
        if not dest.exists() and not candidates:missing.append(item['path']);continue
        for source in candidates:
            assert source.stat().st_size==expected,(source,expected)
            if dest.exists():assert sha256(dest)==sha256(source),f'Different existing artifact {dest}'
            else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        assert dest.stat().st_size==expected,(dest,expected)
        rows.append(dict(path=item['path'],bytes=expected,sha256=sha256(dest),drive_id=item['id']))
    report=dict(status='complete' if not missing else 'partial',files=rows,missing=missing,
        expected_files=len(inventory),copied_files=len(rows),bytes=sum(x['bytes'] for x in rows),
        integrity='Drive inventory size checks; local SHA256 recorded; selected/epoch prediction hashes verified separately')
    atomic_json(ROOT/'reports/common_epoch_20261004/colab_backup_manifest.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['files','missing']},indent=2))

if __name__=='__main__':main()
