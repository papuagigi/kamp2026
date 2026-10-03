"""Restore portable dataset links from a packaged catalog, without changing source files."""
import json
import os
import shutil
from pathlib import Path
from xray_config import ROOT


def main():
    catalog=json.loads((ROOT/'data/xray_roadmap_20261002/variants_catalog.json').read_text())
    made=0
    for record in catalog:
        src=ROOT/record['source'];dst=ROOT/record['destination']
        if not src.is_file():raise FileNotFoundError(src)
        if dst.exists():
            if dst.read_bytes()!=src.read_bytes():raise RuntimeError(f'Existing variant differs: {dst}')
            continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        try:dst.symlink_to(os.path.relpath(src,dst.parent))
        except OSError:shutil.copyfile(src,dst)
        made+=1
    print('Dataset links restored:',made)

if __name__=='__main__':main()
