"""Read-only content/split audit for the actual mixed training dataset."""
from collections import Counter,defaultdict
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
from xray_config import ROOT,DATA
from xray_check_dataset import check_dataset
from xray_recovery import atomic_json,sha256


def main():
    baseline=check_dataset()
    directory=ROOT/'data/xray_combined_20261003'
    rows=json.loads((directory/'provenance.json').read_text())
    inventory=pd.read_csv(ROOT/'data/xray_roadmap_20261002/inventory.csv').set_index('stem')
    official=pd.read_csv(DATA/'manifest.csv',encoding='utf-8-sig').set_index('stem')
    accepted=pd.read_csv(ROOT/'reports/roadmap_20261002/pseudo/decisions.csv')
    accepted=set(accepted.loc[accepted.status=='accepted','stem'])
    seen_pixels={};groups=defaultdict(set);empty=Counter();boxes=Counter();kinds=Counter()
    for r in rows:
        split=r['split'];stem=r['stem'];kind=r['kind']
        # Hash suffixes disambiguate original files with identical names; preserve them.
        parent=stem if kind in ['basic','accepted_pseudo','official_val','official_test'] else stem.removesuffix('__'+kind)
        if kind not in ['basic','accepted_pseudo','official_val','official_test']:assert parent!=stem
        image=directory/'images'/split/(stem+'.png');label=directory/'labels'/split/(stem+'.txt')
        assert sha256(image)==r['image_sha256']==sha256(ROOT/r['source_image'])
        assert sha256(label)==r['label_sha256']==sha256(ROOT/r['source_label'])
        with Image.open(image) as im:rgb=np.asarray(im.convert('RGB'))
        digest=hashlib.sha256(rgb.tobytes()).hexdigest();assert digest==r['pixel_sha256']
        assert digest not in seen_pixels,('Pixel duplicate',stem,seen_pixels.get(digest))
        seen_pixels[digest]=(stem,split)
        assert np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,0],rgb[:,:,2]),('Chromatic pixels remain',stem)
        if split=='train':
            inv=inventory.loc[parent];assert inv.status in ['official','pseudo_candidate']
            assert inv.split not in ['val','test'];groups[int(inv.group)].add('train')
            if kind=='accepted_pseudo':assert parent in accepted and parent not in official.index
            else:assert parent in official.index and official.loc[parent,'split']=='train'
        else:
            assert sha256(image)==sha256(DATA/'images'/split/(stem+'.png'))
            assert sha256(label)==sha256(DATA/'labels'/split/(stem+'.txt'))
            groups[int(official.loc[stem,'burst_id'])].add(split)
        lines=[line for line in label.read_text().splitlines() if line.strip()]
        if not lines:empty[split]+=1
        boxes[split]+=len(lines);kinds[kind]+=1
        for line in lines:
            values=np.array([float(v) for v in line.split()]);assert values.shape==(5,)
            assert values[0]==0 and np.isfinite(values).all()
            assert ((values[1:]>=0)&(values[1:]<=1)).all()
            assert (values[3:]>0).all()
    assert all(len(v)==1 for v in groups.values()),'Capture group leakage'
    counts=dict(Counter(r['split'] for r in rows))
    for split in counts:
        expected={r['stem'] for r in rows if r['split']==split}
        assert expected=={p.stem for p in (directory/'images'/split).glob('*.png')}
        assert expected=={p.stem for p in (directory/'labels'/split).glob('*.txt')}
    result=dict(status='pass',scope='Read-only current data integrity; no model scoring or label semantic proof',
        counts=counts,kinds=dict(kinds),boxes=dict(boxes),empty_label_images=dict(empty),
        real_normal_validation_images=0,real_normal_test_images=0,
        source_and_destination_hash_matches=len(rows),pixel_duplicates=0,cross_split_capture_groups=0,
        remaining_chromatic_pixels_images=0,provenance_sha256=sha256(directory/'provenance.json'),
        fixed_split=baseline,limits=['Grayscale does not prove erased pixels recover the original signal.',
            'Pseudo labels are training estimates; semantic correctness is not proven by hashes.',
            'Removal-generated empty labels are synthetic negatives, not real normal products.',
            'Burst and exact-pixel checks do not prove independence across dates or specimen types.'])
    atomic_json(ROOT/'reports/design_review_20261004/data_integrity.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
