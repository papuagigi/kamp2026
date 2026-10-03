"""Read-only geometry study: equipment proposals, unlabeled groups, image crops.
No training labels are generated and no model is trained or scored.
"""
import json,hashlib,re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from xray_prepare_v2 import RAW,LAB,ROOT,mask_palette
from xray_data_audit import mark_regions
OUT=ROOT/'reports/augmentation_options_study_20261002'


def xyxy(b,w,h):
    return np.array([(b[1]-b[3]/2)*w,(b[2]-b[4]/2)*h,(b[1]+b[3]/2)*w,(b[2]+b[4]/2)*h])


def iou(a,b):
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))


def lab(p):return np.array([list(map(float,l.split())) for l in p.read_text().splitlines() if l.strip()]).reshape(-1,5)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    labels={p.stem:p for p in LAB.glob('*.txt')};seen=set();raw=[];comparisons=[];marked_unmatched=[]
    for p in sorted(RAW.rglob('*.bmp')):
        digest=hashlib.md5(p.read_bytes()).hexdigest()
        if (p.stem,digest) in seen:continue
        seen.add((p.stem,digest));im=Image.open(p);idx=np.asarray(im);h,w=idx.shape
        _,holes,lines=mark_regions(idx)
        name=re.match(r'\d{3}_(\d{8})_(\d{6})\(\d+\)',p.stem)
        r={'stem':p.stem,'path':p.relative_to(ROOT).as_posix(),'md5':digest,'machine':p.relative_to(RAW).parts[0][:3],'dt':datetime.strptime(name[1]+name[2],'%Y%m%d%H%M%S'),'labeled':p.stem in labels,'mark_boxes':len(holes),'mark_lines':lines}
        raw.append(r)
        if p.stem not in labels:continue
        b=lab(labels[p.stem]);eb=[]
        for c in holes:
            x,y,bw,bh=cv2.boundingRect(c);eb.append(np.array([x,y,x+bw,y+bh],float))
        centers=b[:,1:3]*[w,h]
        for k,(bb,center) in enumerate(zip(b,centers)):
            gt=xyxy(bb,w,h);inside=[j for j,c in enumerate(holes) if cv2.pointPolygonTest(c,tuple(center),False)>=0]
            j=max(inside,key=lambda n:iou(gt,eb[n])) if inside else (int(np.argmax([iou(gt,e) for e in eb])) if eb else None)
            match=eb[j] if j is not None else np.full(4,np.nan)
            comparisons.append({'stem':p.stem,'machine':r['machine'],'box_index':k,'inside_equipment_rect':bool(inside),'iou_with_rect_enclosed_area':iou(gt,match) if j is not None else 0.,'equipment_to_gt_area_ratio':float(np.prod(match[2:]-match[:2])/np.prod(gt[2:]-gt[:2])) if inside else None,'gt_width':gt[2]-gt[0],'gt_height':gt[3]-gt[1],'equipment_width':match[2]-match[0],'equipment_height':match[3]-match[1],'gt_x1':gt[0],'gt_y1':gt[1],'gt_x2':gt[2],'gt_y2':gt[3]})
        for j,c in enumerate(holes):
            if not any(cv2.pointPolygonTest(c,tuple(pt),False)>=0 for pt in centers):marked_unmatched.append({'stem':p.stem,'machine':r['machine'],'enclosed_bbox':eb[j].tolist()})
    raw=pd.DataFrame(raw).sort_values(['machine','dt','stem']).reset_index(drop=True)
    gaps=raw.groupby('machine').dt.diff().dt.total_seconds();raw['group']=(gaps.isna()|(gaps>60)).cumsum()
    man=pd.read_csv(ROOT/'data/xray_v2/manifest.csv')
    lm=raw[raw.labeled].merge(man[['stem','split']],on='stem',validate='one_to_one')
    assert (lm.groupby('group').split.nunique()==1).all()
    partitions=lm.groupby('group').split.first().to_dict()
    raw['partition']=raw.group.map(partitions).fillna('new_group')
    unlabeled=raw[~raw.labeled].copy()
    unlabeled['reason']=np.select([unlabeled.mark_boxes.eq(0),unlabeled.partition.isin(['val','test'])],['excluded_no_rect_line_images','excluded_heldout_group'],default='candidate_needs_annotation_review')
    raw.to_csv(OUT/'unique_raw_groups.csv',index=False)
    unlabeled.to_csv(OUT/'unlabeled_inventory.csv',index=False)
    cmp=pd.DataFrame(comparisons);cmp.to_csv(OUT/'equipment_geometry_comparison.csv',index=False)
    (OUT/'equipment_rect_without_txt.json').write_text(json.dumps(marked_unmatched,ensure_ascii=False,indent=2))
    # RandomCrop geometry experiment on TRAIN ONLY; fractions refer to width and height.
    rng=np.random.default_rng(0);crop=[];N=100
    for r in man[man.split=='train'].itertuples():
        bb=lab(ROOT/'data/xray_v2/labels/train'/f'{r.stem}.txt');gt=np.array([xyxy(b,1,1) for b in bb])
        for f in [.5,.7,.8,.9]:
            starts=rng.uniform(0,1-f,(N,2));ends=starts+f
            full=(gt[None,:,:2]>=starts[:,None,:]).all(2)&(gt[None,:,2:]<=ends[:,None,:]).all(2)
            intersects=(np.minimum(gt[None,:,2:],ends[:,None,:])>np.maximum(gt[None,:,:2],starts[:,None,:])).all(2)
            feasible=(gt[:,2:].max(0)-gt[:,:2].min(0)<=f).all()
            crop.append({'stem':r.stem,'crop_side_fraction':f,'trials':N,'n_gt':len(gt),'any_gt_not_fully_retained':int((~full.all(1)).sum()),'zero_visible_gt':int((~intersects.any(1)).sum()),'some_gt_truncated':int((intersects&~full).any(1).sum()),'safe_all_bbox_crop_feasible':bool(feasible)})
    cd=pd.DataFrame(crop);cd.to_csv(OUT/'random_crop_geometry_trials.csv',index=False)
    csum=cd.groupby('crop_side_fraction').agg(trials=('trials','sum'),any_gt_not_fully_retained=('any_gt_not_fully_retained','sum'),zero_visible_gt=('zero_visible_gt','sum'),some_gt_truncated=('some_gt_truncated','sum'),safe_all_bbox_crop_feasible_images=('safe_all_bbox_crop_feasible','sum')).reset_index()
    csum['any_gt_not_fully_retained_pct']=csum.any_gt_not_fully_retained/csum.trials*100
    csum.to_csv(OUT/'crop_geometry_summary.csv',index=False)
    inc=cmp[cmp.inside_equipment_rect]
    summary={'unique_raw_images':len(raw),'labeled_images':int(raw.labeled.sum()),'unlabeled_images':len(unlabeled),'groups_all':int(raw.group.nunique()),'groups_with_official_labels':int(lm.group.nunique()),'unlabeled_by_partition_and_marks':unlabeled.groupby(['partition','mark_boxes']).size().reset_index(name='images').to_dict('records'),'unlabeled_groups':int(unlabeled.group.nunique()),'new_unlabeled_only_groups':int(unlabeled[unlabeled.partition=='new_group'].group.nunique()),'unlabeled_candidate_groups':int(unlabeled[unlabeled.reason=='candidate_needs_annotation_review'].group.nunique()),'unlabeled_by_reason':unlabeled.reason.value_counts().to_dict(),'official_boxes':len(cmp),'official_centers_in_equipment_rect':int(cmp.inside_equipment_rect.sum()),'official_centers_outside_equipment_rect':int((~cmp.inside_equipment_rect).sum()),'equipment_rects_without_txt_center':marked_unmatched,'enclosed_equipment_rect_iou_median':float(cmp.iou_with_rect_enclosed_area.median()),'enclosed_equipment_rect_iou_ge_05':int((cmp.iou_with_rect_enclosed_area>=.5).sum()),'enclosed_equipment_to_gt_area_ratio_median':float(inc.equipment_to_gt_area_ratio.median()),'matched_iou_q10_q50_q90':inc.iou_with_rect_enclosed_area.quantile([.1,.5,.9]).to_dict(),'train_images_in_crop_study':int((man.split=='train').sum()),'random_crop_study':csum.to_dict('records'),'scope':'geometry and inventory only; no detector training or F1/AP experiment; equipment regions are weak proposals, not new ground truth'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
