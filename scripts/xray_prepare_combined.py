"""Materialize the audited union once, preserving official validation/test files."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image
import yaml
from xray_config import ROOT, DATA, V2_SPLIT_MD5

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    out = ROOT / 'data/xray_combined_20261003'
    report = ROOT / 'reports/overnight_20261003'
    assert hashlib.md5((DATA/'split.csv').read_bytes()).hexdigest() == V2_SPLIT_MD5
    records = json.loads((ROOT/'reports/roadmap_20261002/combined_data_union_fingerprints.json').read_text())
    assert len(records) == 2453
    variants = ROOT/'data/xray_roadmap_20261002/variants'
    sources = {}
    for variant in variants.iterdir():
        for im in (variant/'images/train').glob('*.png'):
            label = variant/'labels/train'/f'{im.stem}.txt'
            key = im.resolve()
            if key in sources:
                assert sha(sources[key]) == sha(label), f'Conflicting labels: {im}'
            sources[key] = label.resolve()
    rows, hashes = [], {}
    for split in ['train', 'val', 'test']:
        items = records if split == 'train' else [dict(image=str(p.relative_to(ROOT)),kind='official_'+split) for p in sorted((DATA/'images'/split).glob('*.png'))]
        for record in items:
            im = ROOT/record['image']
            label = sources[im.resolve()] if split == 'train' else DATA/'labels'/split/f'{im.stem}.txt'
            # The earlier audit used a different label serialization for its hash.
            # Verify the actual TXT bytes across variants and copied files here.
            pixel_sha = hashlib.sha256(Image.open(im).convert('RGB').tobytes()).hexdigest()
            assert pixel_sha not in hashes, f'Image duplicate across union/splits: {im}'
            hashes[pixel_sha] = split
            for kind, src in [('images',im),('labels',label)]:
                dst = out/kind/split/src.name
                dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():
                    assert sha(dst) == sha(src), f'Existing file differs: {dst}'
                else:
                    shutil.copyfile(src,dst)
            rows.append(dict(stem=im.stem,split=split,kind=record['kind'],source_image=record['image'],source_label=str(label.relative_to(ROOT)),image_sha256=sha(im),label_sha256=sha(label),pixel_sha256=pixel_sha))
    counts = dict(collections.Counter(r['split'] for r in rows))
    assert counts == {'train':2453,'val':107,'test':97}, counts
    (out/'data.yaml').write_text(yaml.safe_dump(dict(path='.',train='images/train',val='images/val',test='images/test',names={0:'defect'})))
    (out/'provenance.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    report.mkdir(parents=True,exist_ok=True)
    result = dict(dataset=str(out.relative_to(ROOT)),counts=counts,train_kinds=dict(collections.Counter(r['kind'] for r in rows if r['split']=='train')),split_md5=V2_SPLIT_MD5,pixel_duplicates=0,label_conflicts=0,provenance_sha256=sha(out/'provenance.json'))
    (report/'data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
