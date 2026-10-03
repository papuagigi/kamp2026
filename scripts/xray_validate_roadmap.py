"""Check transformed labels, train-only ancestry, and unchanged held-out images."""
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
from xray_config import ROOT,DATA,V2_SPLIT_MD5
from xray_prepare_roadmap import OUT,labels,xyxy


def main():
    inv=pd.read_csv(OUT/'inventory.csv');aug=pd.read_csv(OUT/'augmentation.csv')
    man=pd.read_csv(DATA/'manifest.csv');train=set(man.loc[man.split=='train','stem'])
    assert inv.stem.is_unique and set(aug.parent)<=train and aug.stem.is_unique
    assert set(inv.loc[inv.status=='pseudo_candidate','group']).isdisjoint(set(man.loc[man.split!='train','burst_id']))
    assert hashlib.md5((DATA/'split.csv').read_bytes()).hexdigest()==V2_SPLIT_MD5
    counts={}
    for r in aug.itertuples():
        im=Image.open(ROOT/r.image);w,h=im.size;b=labels(ROOT/r.label);parent=labels(DATA/'labels/train'/f'{r.parent}.txt')
        assert len(b)==r.n_boxes and (b[:,0]==0).all()
        box=xyxy(b,w,h);assert np.isfinite(box).all() and (box[:,:2]>=-1e-6).all() and (box[:,2:]<=[w+1e-6,h+1e-6]).all()
        details=json.loads(r.details)
        original=xyxy(parent,w,h)
        if r.kind=='crop':
            ox,oy=details['origin'];cw,ch=details['size']
            reconstructed=box*np.tile([cw/w,ch/h],2)+np.tile([ox,oy],2)
            assert np.allclose(reconstructed,original,atol=1e-7)
        elif r.kind.startswith('rotate'):
            mat=np.array(details['affine']);corners=np.array([[[x,y],[xx,y],[xx,yy],[x,yy]] for x,y,xx,yy in original])
            transformed=np.concatenate([corners,np.ones((*corners.shape[:2],1))],2)@mat.T
            expected=np.concatenate([transformed.min(1),transformed.max(1)],1)
            assert np.allclose(box,expected,atol=1e-7)
        elif r.kind.startswith('remove'):
            keep=[i for i in range(len(parent)) if i not in details['removed_indices']]
            assert np.allclose(b,parent[keep],atol=1e-10)
            assert details['passed']
        elif r.kind=='offbar':assert len(b)==len(parent)+1 and np.allclose(b[:-1],parent)
        counts[r.kind]=counts.get(r.kind,0)+1
    teacher_verified=False
    for variant in (OUT/'variants').iterdir():
        if not variant.is_dir():continue
        if variant.name=='teacher':
            spec=json.loads((ROOT/'reports/roadmap_20261002/teacher_split.json').read_text())
            fit=set(spec['fit_stems']);cal=set(spec['calibration_stems'])
            assert fit.isdisjoint(cal) and fit|cal <= train
            assert set(man.loc[man.stem.isin(fit),'burst_id']).isdisjoint(man.loc[man.stem.isin(cal),'burst_id'])
            for split,stems in [('train',fit),('val',cal)]:
                images=list((variant/'images'/split).glob('*.png'))
                assert {p.stem for p in images}==stems
                for p in images:
                    assert p.read_bytes()==(DATA/'images/train'/p.name).read_bytes()
                    assert (variant/'labels'/split/(p.stem+'.txt')).read_bytes()==(DATA/'labels/train'/(p.stem+'.txt')).read_bytes()
            teacher_verified=True
            continue
        for split in ['val','test']:
            paths=list((variant/'images'/split).glob('*.png'))
            assert {p.stem for p in paths}==set(man.loc[man.split==split,'stem'])
            for p in paths:
                assert p.read_bytes()==(DATA/'images'/split/p.name).read_bytes()
                assert (variant/'labels'/split/(p.stem+'.txt')).read_bytes()==(DATA/'labels'/split/(p.stem+'.txt')).read_bytes()
    pseudo_count=0
    accepted_path=ROOT/'reports/roadmap_20261002/pseudo/accepted.csv'
    if accepted_path.exists():
        accepted=pd.read_csv(accepted_path);pseudo_count=len(accepted)
        assert set(accepted.stem)<=set(inv.loc[inv.status=='pseudo_candidate','stem'])
        assert set(accepted.group).isdisjoint(set(man.loc[man.split!='train','burst_id']))
        for r in accepted.itertuples():
            with Image.open(ROOT/r.image) as im:w,h=im.size
            b=labels(ROOT/r.label);box=xyxy(b,w,h)
            assert len(b)==r.n_boxes and len(b)>0 and (b[:,0]==0).all()
            assert np.isfinite(box).all() and (box[:,:2]>=0).all() and (box[:,2:]<=[w,h]).all()
    result={'status':'passed','unique_inventory_images':len(inv),'augmented_images_checked':len(aug),
            'teacher_internal_split_verified':teacher_verified,'accepted_pseudo_images_checked':pseudo_count,
            'kinds':counts,'heldout_bytes_equal':True,'train_only_parentage':True,'crop_and_rotation_coordinates_verified':True,
            'synthetic_image_quality_note':'software/geometry checks; not proof of physical realism or normal-product representativeness'}
    path=ROOT/'reports/roadmap_20261002/preprocessing_validation.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
