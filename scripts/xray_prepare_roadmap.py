"""Build isolated, audited preprocessing/ablation datasets; never edit source data."""
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from xray_config import ROOT, DATA, V2_SPLIT_MD5
from xray_check_dataset import check_dataset
from xray_prepare_v2 import load_palette, mask_palette
from xray_eval import product_mask

OUT = ROOT/'data/xray_roadmap_20261002'


def labels(path):
    return np.array([list(map(float, l.split())) for l in path.read_text().splitlines() if l.strip()]).reshape(-1, 5)


def xyxy(b, w, h):
    return np.column_stack(((b[:, 1:3]-b[:, 3:5]/2)*[w,h],
                            (b[:, 1:3]+b[:, 3:5]/2)*[w,h]))


def norm(boxes, w, h):
    return np.column_stack((np.zeros(len(boxes)), (boxes[:,:2]+boxes[:,2:])/2/[w,h],
                            (boxes[:,2:]-boxes[:,:2])/[w,h]))


def put_label(path, b):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join('0 '+' '.join(f'{v:.12f}' for v in row[1:])+'\n' for row in b))


def link(src, dst):
    import os
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(os.path.relpath(src, dst.parent))


def removal(gray, b, remove):
    h,w=gray.shape; boxes=xyxy(b,w,h); mask=np.zeros_like(gray)
    for i in remove:
        x1,y1,x2,y2=boxes[i]
        cv2.rectangle(mask,(max(0,int(np.floor(x1))-2),max(0,int(np.floor(y1))-2)),
                      (min(w-1,int(np.ceil(x2))+2),min(h-1,int(np.ceil(y2))+2)),255,-1)
    keep=[i for i in range(len(b)) if i not in remove]
    for i in keep:
        x1,y1,x2,y2=boxes[i].astype(int)
        if mask[max(0,y1):min(h,y2+1),max(0,x1):min(w,x2+1)].any():
            return None,None,{'passed':False,'reason':'mask_overlaps_retained_label'}
    img=cv2.inpaint(gray,mask,3,cv2.INPAINT_NS)
    assert np.array_equal(gray[mask==0],img[mask==0])
    residual=[]
    # Conservative local residual check; passing is not proof of real normality.
    for i in remove:
        cx,cy=np.round(b[i,1:3]*[w,h]).astype(int)
        roi=img[max(0,cy-6):cy+7,max(0,cx-6):cx+7]
        core=img[max(0,cy-1):cy+2,max(0,cx-1):cx+2]
        residual.append(float(np.median(roi))-float(core.min()))
    passed=max(residual,default=0)<=12
    return img,b[keep],{'passed':passed,'residual_darkness':residual,'removed_indices':list(remove),
                         'pixels':int((mask>0).sum()),'reason':'' if passed else 'dark_residual_over_12'}


def main():
    check_dataset()
    if OUT.exists(): raise RuntimeError(f'Existing output preserved: {OUT}')
    inventory=pd.read_csv(ROOT/'reports/augmentation_options_study_20261002/unique_raw_groups.csv')
    unlab=pd.read_csv(ROOT/'reports/augmentation_options_study_20261002/unlabeled_inventory.csv')
    man=pd.read_csv(DATA/'manifest.csv').fillna('')
    assert len(inventory)==2532 and not inventory.duplicated(['stem','md5']).any()
    duplicate_names=set(inventory.loc[inventory.stem.duplicated(False),'stem'])
    partitions=man.set_index('burst_id').split.to_dict()
    assert all(partitions.get(int(r.group),'new_group')==r.partition for r in inventory.itertuples())
    OUT.mkdir(parents=True)
    (OUT/'images/clean').mkdir(parents=True)
    rng=np.random.default_rng(0); records=[]; audit=[]
    candidate=set(unlab.loc[unlab.reason=='candidate_needs_annotation_review',['stem','md5']].itertuples(index=False,name=None))
    for r in inventory.itertuples():
        p=ROOT/r.path
        image_id=f'{r.stem}__{r.md5[:8]}' if r.stem in duplicate_names else r.stem
        assert hashlib.md5(p.read_bytes()).hexdigest()==r.md5
        idx=load_palette(p)
        if r.mark_boxes==0:
            records.append(dict(stem=image_id,original_stem=r.stem,source=r.path,group=r.group,split=r.partition,
                                status='excluded_position_anomaly',image='',masked_px=0))
            continue
        clean,n=mask_palette(idx)
        assert np.array_equal(clean[idx<244],idx[idx<244])
        out=OUT/'images/clean'/f'{image_id}.png'; assert cv2.imwrite(str(out),clean)
        status='official' if r.labeled else ('pseudo_candidate' if (r.stem,r.md5) in candidate else 'excluded_heldout_group')
        records.append(dict(stem=image_id,original_stem=r.stem,source=r.path,group=r.group,split=r.partition,
                            status=status,image=str(out.relative_to(ROOT)),masked_px=n))
        if r.labeled:
            split=man.set_index('stem').loc[r.stem,'split']
            previous=cv2.imread(str(DATA/'images'/split/f'{r.stem}.png'),0)
            assert np.array_equal(previous,clean)
    pd.DataFrame(records).to_csv(OUT/'inventory.csv',index=False)
    aug=[]

    def save(r,kind,img,b,details):
        h,w=img.shape; bb=xyxy(b,w,h)
        assert np.isfinite(b).all() and (bb[:,:2]>=-1e-6).all() and (bb[:,2:]<=[w+1e-6,h+1e-6]).all()
        assert (b[:,3:]>0).all()
        stem=f'{r.stem}__{kind}'; ip=OUT/'images/augment'/f'{stem}.png';lp=OUT/'labels/augment'/f'{stem}.txt'
        ip.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(ip),img);put_label(lp,b)
        aug.append(dict(stem=stem,parent=r.stem,burst_id=r.burst_id,kind=kind,
                        image=str(ip.relative_to(ROOT)),label=str(lp.relative_to(ROOT)),n_boxes=len(b),
                        details=json.dumps(details)))

    for r in man[man.split=='train'].itertuples():
        if '라벨누락' in r.label_issue:
            audit.append(dict(stem=r.stem,kind='all',passed=False,reason='known_missing_official_label'));continue
        g=cv2.imread(str(OUT/'images/clean'/f'{r.stem}.png'),0);h,w=g.shape
        b=labels(DATA/'labels/train'/f'{r.stem}.txt');boxes=xyxy(b,w,h)
        f=float(rng.uniform(.8,.95));cw,ch=int(w*f),int(h*f)
        lo=np.maximum(0,np.ceil(boxes[:,2:].max(0)-[cw,ch])).astype(int)
        hi=np.minimum([w-cw,h-ch],np.floor(boxes[:,:2].min(0))).astype(int)
        if (hi>=lo).all():
            ox,oy=[int(rng.integers(lo[k],hi[k]+1)) for k in range(2)]
            cropped=cv2.resize(g[oy:oy+ch,ox:ox+cw],(w,h),interpolation=cv2.INTER_LINEAR)
            cb=(boxes-np.tile([ox,oy],2))*np.tile([w/cw,h/ch],2)
            save(r,'crop',cropped,norm(cb,w,h),dict(origin=[ox,oy],size=[cw,ch],all_boxes_retained=True))
        for angle in [-7.5,7.5]:
            m=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
            corners=np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for x1,y1,x2,y2 in boxes])
            pts=np.concatenate([corners,np.ones((*corners.shape[:2],1))],2)@m.T
            rb=np.column_stack([pts.min(1),pts.max(1)])
            if (rb[:,:2]>=0).all() and (rb[:,2:]<=[w,h]).all():
                rotated=cv2.warpAffine(g,m,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float(np.median(g[:5])))
                save(r,'rotate_m' if angle<0 else 'rotate_p',rotated,norm(rb,w,h),dict(angle=angle,affine=m.tolist()))
        for kind,remove in [('remove_all',list(range(len(b)))),('remove_partial',[int(rng.integers(len(b)))])]:
            if kind=='remove_partial' and len(b)<2:continue
            changed,remain,qa=removal(g,b,remove)
            audit.append(dict(stem=r.stem,kind=kind,**qa))
            if qa['passed']:save(r,kind,changed,remain,qa)
        # Exploratory attenuation-like dark residual transfer from a TRAIN image only.
        # This is not claimed to reproduce physical X-ray formation.
        pm=product_mask(g);dist=cv2.distanceTransform(pm,cv2.DIST_L2,3)
        yy,xx=np.indices(g.shape);cx,cy=np.round(b[0,1:3]*[w,h]).astype(int)
        candidates=(dist>18)&(g>np.percentile(g[pm>0],55)) if pm.any() else np.zeros_like(pm,dtype=bool)
        for q in b:candidates &= ((xx-q[1]*w)**2+(yy-q[2]*h)**2)>45**2
        candidates[:15]=False;candidates[-15:]=False;candidates[:,:15]=False;candidates[:,-15:]=False
        positions=np.argwhere(candidates)
        if len(positions) and 8<=cx<w-8 and 8<=cy<h-8:
            ty,tx=positions[int(rng.integers(len(positions)))];patch=g[cy-7:cy+8,cx-7:cx+8].astype(float)
            residual=np.minimum(0,patch-cv2.GaussianBlur(patch,(0,0),2.5))
            taper=np.exp(-((np.arange(-7,8)[:,None]**2+np.arange(-7,8)[None,:]**2)/(2*3**2)))
            residual*=taper
            sg=g.copy();sg[ty-7:ty+8,tx-7:tx+8]=np.clip(sg[ty-7:ty+8,tx-7:tx+8]+residual,0,255).astype('uint8')
            added=b[0].copy();added[1:3]=[tx/w,ty/h]
            save(r,'offbar',sg,np.vstack([b,added]),dict(source_parent=r.stem,target=[int(tx),int(ty)],
                    method='negative_highpass_residual_tapered',physical_realism_validated=False))
    ad=pd.DataFrame(aug);ad.to_csv(OUT/'augmentation.csv',index=False)
    (OUT/'removal_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    choices={'official':[], 'geometry':['crop','rotate_m','rotate_p'],
             'remove_all':['remove_all'],'remove_partial':['remove_partial'],
             'geometry_removal':['crop','rotate_m','rotate_p','remove_all','remove_partial'],
             'offbar':['offbar']}
    variant_stats={}
    for name,kinds in choices.items():
        v=OUT/'variants'/name
        for r in man.itertuples():
            link(OUT/'images/clean'/f'{r.stem}.png',v/'images'/r.split/f'{r.stem}.png')
            link(DATA/'labels'/r.split/f'{r.stem}.txt',v/'labels'/r.split/f'{r.stem}.txt')
        extra=ad[ad.kind.isin(kinds)]
        for r in extra.itertuples():
            link(ROOT/r.image,v/'images/train'/f'{r.stem}.png');link(ROOT/r.label,v/'labels/train'/f'{r.stem}.txt')
        (v/'data.yaml').write_text(yaml.safe_dump({'path':'.','train':'images/train','val':'images/val','test':'images/test','names':{0:'defect'}},sort_keys=False))
        # The adapters resolve this project-relative path from ROOT.
        variant_stats[name]={'train_images':296+len(extra),'added_images':len(extra),'added_boxes':int(extra.n_boxes.sum())}
    assert hashlib.md5((DATA/'split.csv').read_bytes()).hexdigest()==V2_SPLIT_MD5
    summary={'clean_images':sum(bool(r['image']) for r in records),'statuses':pd.Series([r['status'] for r in records]).value_counts().to_dict(),
             'augmentation_images':ad.kind.value_counts().to_dict(),'variants':variant_stats,
             'removal_quality_passed':sum(bool(a.get('passed')) for a in audit),'removal_quality_rejected':sum(not bool(a.get('passed')) for a in audit),
             'seed':0,'fixed_split_md5':V2_SPLIT_MD5,'masked_official_images_equal_existing':500,
             'negative_images_are_synthetic':True,'all_new_labels_from_training_parents_only':True}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
