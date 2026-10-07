"""Read-only file, split and augmentation-coordinate audit of the trained mixture."""
import hashlib
import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from xray_config import ROOT, DATA, V2_SPLIT_MD5
from xray_check_dataset import check_dataset

OUT = ROOT / 'reports/direct_analysis_20261007'
MIX = ROOT / 'data/xray_direct_training_20261006'


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def boxes(p):
    return np.array([list(map(float, x.split())) for x in p.read_text().splitlines() if x.strip()]).reshape(-1, 5)


def corners(b, w, h):
    c, s = b[:, 1:3] * [w, h], b[:, 3:5] * [w, h]
    return np.c_[c-s/2, c+s/2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    official = check_dataset()
    rows = json.loads((MIX/'provenance.json').read_text())
    man = pd.DataFrame(rows)
    excluded = set(json.loads((ROOT/'reports/direct_training_20261006/excluded_photos.json').read_text())['stems'])
    assert man.stem.is_unique
    assert not (man.groupby('group').split.nunique() > 1).any()
    assert not set(man.parent) & excluded
    assert sha(MIX/'provenance.json') == json.loads((MIX/'summary.json').read_text())['provenance_sha256']
    for split in ['train', 'val', 'test']:
        names = set(man.loc[man.split.eq(split), 'stem'])
        for kind, suffix in [('images', 'png'), ('labels', 'txt')]:
            assert {p.stem for p in (MIX/kind/split).glob('*.'+suffix)} == names
    sources = {}; pixels = {}; stats = []; errors = []; geometry = Counter(); samples = Counter()
    for i, r in enumerate(rows):
        ip=MIX/'images'/r['split']/(r['stem']+'.png'); lp=MIX/'labels'/r['split']/(r['stem']+'.txt')
        assert sha(ip)==r['image_sha256'] and sha(lp)==r['label_sha256'], r['stem']
        raw=cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        assert raw is not None
        if raw.ndim==3:
            assert np.all(raw[:,:,0]==raw[:,:,1]) and np.all(raw[:,:,1]==raw[:,:,2]), 'colored input'
        image=cv2.imread(str(ip),0); h,w=image.shape; b=boxes(lp); xy=corners(b,w,h)
        assert len(b)==r['n_boxes'] and np.isfinite(b).all() and (b[:,0]==0).all()
        assert (b[:,3:]>0).all() and (xy[:,:2]>=-1e-4).all() and (xy[:,2:]<=[w+1e-4,h+1e-4]).all()
        pixel=hashlib.sha256(image.tobytes()).hexdigest(); assert pixel==r['pixel_sha256']
        assert pixel not in pixels, ('duplicate pixel',r['stem'],pixels.get(pixel)); pixels[pixel]=r['stem']
        if r['parent'] not in sources:
            sb=boxes(ROOT/r['source_label']); sim=cv2.imread(str(ROOT/r['source_image']),0)
            sources[r['parent']] = (sb,sim.shape)
        sb,shape=sources[r['parent']]; sh,sw=shape; sxy=corners(sb,sw,sh); d=r['details']; kind=r['kind']
        expected=None
        if kind in ['official','direct_visual','official_val','official_test']:
            assert sha(ip)==sha(ROOT/r['source_image']) and sha(lp)==sha(ROOT/r['source_label'])
            expected=sxy
        elif kind=='crop':
            ox,oy=d['origin'];cw,ch=d['size']; expected=(sxy-[ox,oy,ox,oy])*[w/cw,h/ch,w/cw,h/ch]
        elif kind in ['rotate_m','rotate_p']:
            m=np.asarray(d['affine']); c=np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for x1,y1,x2,y2 in sxy])
            pts=np.concatenate([c,np.ones((*c.shape[:2],1))],2)@m.T
            expected=np.c_[pts.min(1),pts.max(1)]
        elif kind=='remove_all':
            assert len(b)==0 and d['passed']
        elif kind=='remove_partial':
            assert len(b)==len(sb)-1 and d['passed']
            assert all(np.min(np.max(np.abs(sxy-x),axis=1))<1e-4 for x in xy)
        elif kind=='offbar':
            assert len(b)==len(sb)+1
            assert np.max(np.abs(xy[:-1]-sxy))<1e-4
            assert np.max(np.abs(b[-1,1:3]*[w,h]-d['target']))<1e-4
            assert np.max(np.abs(b[-1,3:]-sb[0,3:]))<1e-7
        if expected is not None:
            error=float(np.max(np.abs(xy-expected))) if len(xy) else 0.
            assert error<1e-4,(r['stem'],error);errors.append(error)
        geometry[kind]+=1
        # Reproduce a fixed small image sample; all other images have full hash checks.
        if kind in ['crop','rotate_m','rotate_p'] and samples[kind]<3:
            sim=cv2.imread(str(ROOT/r['source_image']),0)
            if kind=='crop':
                recreated=cv2.resize(sim[oy:oy+ch,ox:ox+cw],(w,h),interpolation=cv2.INTER_LINEAR)
            else:
                recreated=cv2.warpAffine(sim,m,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float(np.median(sim[:5])))
            assert np.array_equal(image,recreated);samples[kind]+=1
        if kind in ['official','direct_visual','official_val','official_test']:
            for j,x in enumerate(b):
                stats.append(dict(stem=r['stem'],split=r['split'],origin=r['label_origin'],box=j,w=x[3]*w,h=x[4]*h,long_side=max(x[3]*w,x[4]*h)))
        if (i+1)%3000==0:print('audited',i+1,flush=True)
    table=pd.DataFrame(stats);table.to_csv(OUT/'label_dimensions.csv',index=False)
    dist=table.groupby(['split','origin']).long_side.agg(['count','median','min','max']).reset_index().to_dict('records')
    result=dict(status='passed',official_dataset=official,files_hashed=len(rows)*2,images=len(rows),
        split_counts=man.split.value_counts().to_dict(),train_kinds=man[man.split.eq('train')].kind.value_counts().to_dict(),
        parent_count=man[man.split.eq('train')].parent.nunique(),group_overlap=0,pixel_duplicates=0,
        excluded_ambiguous_included=0,all_inputs_grayscale=True,all_label_coordinates_valid=True,
        geometry_rows=dict(geometry),max_coordinate_reconstruction_error_px=max(errors),pixel_reconstruction_sample=dict(samples),
        label_long_side_summary=dist,provenance_sha256=sha(MIX/'provenance.json'),
        limitation='Geometry and lineage audit, not independently measured annotation accuracy or physical realism of synthesis.')
    (OUT/'data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
