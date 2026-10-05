"""Build training-only role datasets, retaining full official validation unchanged."""
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
import yaml
from PIL import Image
from xray_config import ROOT, DATA, V2_SPLIT_MD5
from xray_review_geometry import around, tiles, clipped_labels
from xray_recovery import atomic_json, sha256

OUT = ROOT/'data/xray_review_roles_20261005'


def main():
    assert hashlib.md5((DATA/'split.csv').read_bytes()).hexdigest() == V2_SPLIT_MD5
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = pd.read_csv(DATA/'manifest.csv')
    train = manifest[(manifest.split == 'train') & manifest.label_issue.isna()]
    records = []
    for role in ['candidate', 'miss']:
        directory = OUT/role
        for kind in ['images', 'labels']:
            for split in ['train', 'val']:
                (directory/kind/split).mkdir(parents=True)
        for row in train.itertuples():
            path = DATA/'images/train'/f'{row.stem}.png'
            with Image.open(path) as source:
                im = source.convert('RGB')
            values = [list(map(float, line.split())) for line in
                      (DATA/'labels/train'/f'{row.stem}.txt').read_text().splitlines() if line.strip()]
            boxes = [[v[1]*im.width, v[2]*im.height, v[3]*im.width, v[4]*im.height] for v in values]
            if role == 'candidate':
                windows = [around(x, y, im.width, im.height) for x,y,_,_ in boxes]
                # At most one non-overlapping background crop per parent.
                for window in tiles(im.width, im.height, 96, 16):
                    labels, _ = clipped_labels(boxes, window)
                    if not labels:
                        windows.append(window)
                        break
            else:
                windows = tiles(im.width, im.height)
            for index, window in enumerate(dict.fromkeys(windows)):
                labels, partial = clipped_labels(boxes, window)
                if partial:
                    continue  # Do not teach a truncated foreign object as empty background.
                name = f'{row.stem}__{role}_{index:02d}'
                im.crop(window).save(directory/'images/train'/f'{name}.png')
                (directory/'labels/train'/f'{name}.txt').write_text(
                    ''.join('0 '+' '.join(f'{v:.9f}' for v in label[1:])+'\n' for label in labels))
                records.append(dict(role=role,stem=name,parent=row.stem,burst_id=str(row.burst_id),
                    parent_sha256=sha256(path),window=window,boxes=len(labels),
                    image_sha256=sha256(directory/'images/train'/f'{name}.png')))
        for kind, suffix in [('images','.png'), ('labels','.txt')]:
            for source in sorted((DATA/kind/'val').glob('*'+suffix)):
                target = directory/kind/'val'/source.name
                target.symlink_to(os.path.relpath(source,target.parent))
        (directory/'data.yaml').write_text(yaml.safe_dump(dict(path=str(directory),train='images/train',
            val='images/val',names={0:'defect'}),sort_keys=False))
    counts = {role:dict(images=sum(x['role']==role for x in records),
               boxes=sum(x['boxes'] for x in records if x['role']==role),
               empty=sum(x['boxes']==0 for x in records if x['role']==role)) for role in ['candidate','miss']}
    atomic_json(OUT/'manifest.json',dict(split_md5=V2_SPLIT_MD5,counts=counts,training_parents=len(train),
        excluded_train_parents=manifest[(manifest.split=='train') & manifest.label_issue.notna()].stem.tolist(),
        records=records,notes=['Only official training parents; no pseudo-labels or test images.',
        'Background labels follow official TXT; not certified normal factory products.',
        'Training candidate crops use GT; validation runtime uses model A boxes only.',
        'Validation is reused development data, not a new independent test.']))
    print(json.dumps(counts),flush=True)

if __name__=='__main__':
    main()
