"""Build the approved direct-label mixture; augment training parents only."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml

from xray_config import ROOT, DATA, V2_SPLIT_MD5
from xray_eval import product_mask
from xray_prepare_roadmap import labels, xyxy, norm, put_label
from xray_refine_removal import refine

OUT = ROOT / 'data/xray_direct_training_20261006'
REPORT = ROOT / 'reports/direct_training_20261006'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert hashlib.md5((DATA/'split.csv').read_bytes()).hexdigest() == V2_SPLIT_MD5
    if OUT.exists():
        raise FileExistsError(f'Preserve existing dataset: {OUT}')
    official = list(csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig')))
    direct = list(csv.DictReader((ROOT/'data/xray_direct_labels_20261006/manifest.csv').open()))
    excluded = set(json.loads((REPORT/'excluded_photos.json').read_text())['stems'])
    inventory = {r['stem']: r for r in csv.DictReader((ROOT/'data/xray_roadmap_20261002/inventory.csv').open())}
    parents = [dict(stem=r['stem'], image=f'data/xray_v2/images/train/{r["stem"]}.png',
                    label=f'data/xray_v2/labels/train/{r["stem"]}.txt', group=int(r['burst_id']),
                    origin='official', label_issue=r['label_issue']) for r in official if r['split']=='train']
    parents += [dict(stem=r['stem'], image=r['image'], label=r['label'], group=int(r['group']),
                     origin='direct_visual', label_issue='') for r in direct
                if r['export_status']=='reviewed_training_candidate' and r['stem'] not in excluded]
    assert len(parents)==2172 and len(excluded)==24
    heldout={int(r['burst_id']) for r in official if r['split'] in ['val','test']}
    assert not {r['group'] for r in parents} & heldout
    assert len({r['stem'] for r in parents})==2172
    for split in ['train','val','test']:
        for kind in ['images','labels']:(OUT/kind/split).mkdir(parents=True)
    rows=[];checks=[];pixels={};rng=np.random.default_rng(0)

    def save(parent, kind, image, boxes, details, split='train', original_image=None, original_label=None):
        h,w=image.shape;bb=xyxy(boxes,w,h)
        assert np.isfinite(boxes).all() and (bb[:,:2]>=-1e-6).all() and (bb[:,2:]<=[w+1e-6,h+1e-6]).all()
        assert (boxes[:,3:]>0).all() and (boxes[:,0]==0).all()
        stem=parent['stem'] if kind in ['official','direct_visual','official_val','official_test'] else parent['stem']+'__'+kind
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        if digest in pixels:
            assert pixels[digest]['split']==split, 'Pixel duplicate leaks across split'
            assert original_image is None, 'Duplicate real parent'
            checks.append(dict(parent=parent['stem'],kind=kind,passed=False,reason='duplicate_pixels'))
            return
        ip=OUT/'images'/split/(stem+'.png');lp=OUT/'labels'/split/(stem+'.txt')
        if original_image:
            shutil.copyfile(original_image,ip);shutil.copyfile(original_label,lp)
        else:
            assert cv2.imwrite(str(ip),image);put_label(lp,boxes)
        record=dict(stem=stem,parent=parent['stem'],split=split,kind=kind,group=parent['group'],
                    label_origin=parent['origin'],n_boxes=len(boxes),source_image=parent['image'],
                    source_label=parent['label'],image_sha256=sha(ip),label_sha256=sha(lp),
                    pixel_sha256=digest,details=details)
        rows.append(record);pixels[digest]=record

    for count,r in enumerate(parents,1):
        g=cv2.imread(str(ROOT/r['image']),0);h,w=g.shape;b=labels(ROOT/r['label']);boxes=xyxy(b,w,h)
        assert len(b)>0 and r['stem'] not in excluded
        save(r,r['origin'],g,b,{},original_image=ROOT/r['image'],original_label=ROOT/r['label'])
        if '라벨누락' in r['label_issue']:
            checks.append(dict(parent=r['stem'],kind='all',passed=False,reason='known_missing_official_label'));continue
        # All annotated objects remain inside the crop; enlarge to original size.
        f=float(rng.uniform(.8,.95));cw,ch=int(w*f),int(h*f)
        lo=np.maximum(0,np.ceil(boxes[:,2:].max(0)-[cw,ch])).astype(int)
        hi=np.minimum([w-cw,h-ch],np.floor(boxes[:,:2].min(0))).astype(int)
        if (hi>=lo).all():
            ox,oy=[int(rng.integers(lo[k],hi[k]+1)) for k in range(2)]
            img=cv2.resize(g[oy:oy+ch,ox:ox+cw],(w,h),interpolation=cv2.INTER_LINEAR)
            cb=(boxes-np.tile([ox,oy],2))*np.tile([w/cw,h/ch],2)
            save(r,'crop',img,norm(cb,w,h),dict(origin=[ox,oy],size=[cw,ch],all_boxes_retained=True))
        for angle in [-7.5,7.5]:
            m=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
            corners=np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for x1,y1,x2,y2 in boxes])
            pts=np.concatenate([corners,np.ones((*corners.shape[:2],1))],2)@m.T
            rb=np.column_stack([pts.min(1),pts.max(1)])
            if (rb[:,:2]>=0).all() and (rb[:,2:]<=[w,h]).all():
                img=cv2.warpAffine(g,m,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float(np.median(g[:5])))
                save(r,'rotate_m' if angle<0 else 'rotate_p',img,norm(rb,w,h),dict(angle=angle,affine=m.tolist()))
        for kind,remove in [('remove_all',list(range(len(b)))),('remove_partial',[int(rng.integers(len(b)))])]:
            if kind=='remove_partial' and len(b)<2:continue
            img,kept,qa=refine(g,b,remove);checks.append(dict(parent=r['stem'],kind=kind,**qa))
            if qa['passed']:save(r,kind,img,kept,qa)
        # Copy the local attenuation-shaped residual within the product, away from labels.
        pm=product_mask(g);distance=cv2.distanceTransform(pm,cv2.DIST_L2,3);yy,xx=np.indices(g.shape)
        choices=(distance>18)&(g>np.percentile(g[pm>0],55)) if pm.any() else np.zeros_like(pm,dtype=bool)
        for q in b:choices &= ((xx-q[1]*w)**2+(yy-q[2]*h)**2)>45**2
        choices[:15]=False;choices[-15:]=False;choices[:,:15]=False;choices[:,-15:]=False
        positions=np.argwhere(choices);cx,cy=np.round(b[0,1:3]*[w,h]).astype(int)
        if len(positions) and 8<=cx<w-8 and 8<=cy<h-8:
            ty,tx=positions[int(rng.integers(len(positions)))];patch=g[cy-7:cy+8,cx-7:cx+8].astype(float)
            residual=np.minimum(0,patch-cv2.GaussianBlur(patch,(0,0),2.5))
            residual*=np.exp(-((np.arange(-7,8)[:,None]**2+np.arange(-7,8)[None,:]**2)/(2*3**2)))
            img=g.copy();img[ty-7:ty+8,tx-7:tx+8]=np.clip(img[ty-7:ty+8,tx-7:tx+8]+residual,0,255).astype('uint8')
            peak=int((g.astype(int)-img.astype(int)).max())
            if peak>=5:
                added=b[0].copy();added[1:3]=[tx/w,ty/h]
                save(r,'offbar',img,np.vstack([b,added]),dict(source_parent=r['stem'],source_center=[int(cx),int(cy)],
                     target=[int(tx),int(ty)],peak_added_darkness=peak,physical_realism_validated=False))
        if count%250==0:print(json.dumps(dict(parents=count,generated=len(rows))),flush=True)
    for split in ['val','test']:
        for r in [r for r in official if r['split']==split]:
            p=dict(stem=r['stem'],image=f'data/xray_v2/images/{split}/{r["stem"]}.png',
                   label=f'data/xray_v2/labels/{split}/{r["stem"]}.txt',group=int(r['burst_id']),origin='official')
            ip=ROOT/p['image'];lp=ROOT/p['label']
            save(p,'official_'+split,cv2.imread(str(ip),0),labels(lp),{},split,ip,lp)
    counts=dict(Counter(r['split'] for r in rows));kinds=dict(Counter(r['kind'] for r in rows if r['split']=='train'))
    assert counts['val']==107 and counts['test']==97
    (OUT/'data.yaml').write_text(yaml.safe_dump(dict(path='.',train='images/train',val='images/val',test='images/test',names={0:'defect'})))
    (OUT/'provenance.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    result=dict(dataset=str(OUT.relative_to(ROOT)),counts=counts,train_kinds=kinds,train_parents=2172,
                direct_photos=1876,official_train=296,excluded_uncertain=24,split_md5=V2_SPLIT_MD5,
                heldout_group_overlap=0,duplicate_images=0,seed=0,synthetic_negatives_are_not_real_normal=True,
                rejected_augmentations=dict(Counter(c['reason'] for c in checks if not c['passed'])),
                provenance_sha256=sha(OUT/'provenance.json'))
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (REPORT/'augmentation_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    (REPORT/'data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
