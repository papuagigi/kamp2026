"""Calibrate two-architecture/flip agreement on held-out TRAIN groups, then label candidates.

Only image-level accepted pseudo labels become training data. Official GT is immutable.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta
import yaml
from xray_config import ROOT,DATA
from xray_model_io import seed_all
from xray_predict_controlled import Predictor
from xray_prepare_roadmap import OUT,labels,xyxy,norm,put_label,link
from xray_prepare_v2 import load_palette
from xray_data_audit import mark_regions
from xray_eval import product_mask

REPORT=ROOT/'reports/roadmap_20261002/pseudo'


def overlaps(a,b):
    if len(a)==0 or len(b)==0:return np.zeros((len(a),len(b)))
    inter=np.maximum(0,np.minimum(a[:,None,2:4],b[None,:,2:4])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    aa=(a[:,2:4]-a[:,:2]).prod(1);bb=(b[:,2:4]-b[:,:2]).prod(1)
    return inter/np.maximum(1e-9,aa[:,None]+bb[None,:]-inter)


def consensus(views,threshold):
    filtered=[np.asarray(v,float).reshape(-1,5) for v in views]
    filtered=[v[v[:,4]>=threshold] for v in filtered]
    if not len(filtered[0]):return None,'no_detection_not_normal'
    if len({len(v) for v in filtered})!=1:return None,'models_or_flip_disagree_count'
    reference=filtered[0];aligned=[reference]
    centers=(reference[:,:2]+reference[:,2:4])/2
    for v in filtered[1:]:
        ov=overlaps(reference,v);ii,jj=linear_sum_assignment(1-ov)
        v=v[jj[np.argsort(ii)]]
        dist=np.linalg.norm(centers-(v[:,:2]+v[:,2:4])/2,axis=1)
        if (dist>4).any() or (np.diag(overlaps(reference,v))<.4).any():return None,'models_or_flip_disagree_location'
        aligned.append(v)
    fused=np.median(np.stack(aligned),axis=0);fused[:,4]=np.min(np.stack(aligned)[:,:,4],axis=0)
    if len(fused)>1:
        ov=overlaps(fused,fused);np.fill_diagonal(ov,0)
        if (ov>.4).any():return None,'duplicate_boxes'
    return fused,''


def pixel_candidates(g,cut):
    bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    binary=((bh>=cut)&(product_mask(g)>0)).astype('uint8')
    n,cc,stats,centers=cv2.connectedComponentsWithStats(binary,8);points=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if 1<=area<=40 and max(w,h)<=10 and max(w,h)/max(1,min(w,h))<=3:
            points.append(centers[i])
    return np.array(points).reshape(-1,2)


def checks(g,fused,holes,pixel_cut):
    centers=(fused[:,:2]+fused[:,2:4])/2
    h,w=g.shape;size=fused[:,2:4]-fused[:,:2]
    if (size<2).any() or (size>25).any() or (fused[:,:2]<0).any() or (fused[:,2:4]>[w,h]).any():return 'unusual_box_size_or_boundary'
    for hole in holes:
        if not any(cv2.pointPolygonTest(hole,tuple(pt),False)>=0 for pt in centers):return 'equipment_candidate_unresolved'
    points=pixel_candidates(g,pixel_cut)
    if len(points) and (np.linalg.norm(points[:,None,:]-centers[None,:,:],axis=2).min(1)>8).any():return 'unexplained_dark_point'
    return ''


def true_complete(fused,gt):
    if len(fused)!=len(gt):return False
    ii,jj=linear_sum_assignment(1-overlaps(fused,gt))
    return bool((overlaps(fused,gt)[ii,jj]>=.5).all())


def predict_views(predictors,path):
    im=Image.open(path).convert('RGB');w,h=im.size;views=[]
    for model in predictors:
        views.append(model(im,.05))
        pred=model(ImageOps.mirror(im),.05)
        if len(pred):
            old=pred[:,[0,2]].copy();pred[:,0]=w-old[:,1];pred[:,2]=w-old[:,0]
        views.append(pred)
    return [v.tolist() for v in views]


def main():
    p=argparse.ArgumentParser();p.add_argument('--yolo-run',default='v2_roadmap_yolo_teacher')
    p.add_argument('--rfdetr-run',default='v2_roadmap_rfdetr_teacher');p.add_argument('--device',default='auto')
    a=p.parse_args();seed_all();REPORT.mkdir(parents=True,exist_ok=True)
    specs=[];models=[]
    for run in [a.yolo_run,a.rfdetr_run]:
        meta=json.loads((ROOT/'runs'/run/'execution.json').read_text());assert meta['status']=='complete' and meta['variant']=='teacher'
        checkpoint=ROOT/meta['checkpoint'];specs.append(dict(run=run,sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        models.append(Predictor(meta['model'],checkpoint,a.device,meta['resolution']))
    provenance=REPORT/'models.json'
    if provenance.exists():assert json.loads(provenance.read_text())==specs
    else:provenance.write_text(json.dumps(specs,indent=2))
    inv=pd.read_csv(OUT/'inventory.csv').set_index('stem');man=pd.read_csv(DATA/'manifest.csv').set_index('stem')
    split=json.loads((REPORT.parent/'teacher_split.json').read_text());calibration=[]

    def cached(stem):
        cache=REPORT/'prediction_cache'/f'{stem}.json';cache.parent.mkdir(exist_ok=True)
        if cache.exists():return json.loads(cache.read_text())
        result=predict_views(models,ROOT/inv.loc[stem,'image']);cache.write_text(json.dumps(result));return result

    for i,stem in enumerate(split['calibration_stems']):
        g=cv2.imread(str(ROOT/inv.loc[stem,'image']),0);h,w=g.shape
        _,holes,_=mark_regions(load_palette(ROOT/inv.loc[stem,'source']))
        calibration.append((stem,cached(stem),g,holes,xyxy(labels(DATA/'labels/train'/f'{stem}.txt'),w,h)))
        if (i+1)%20==0:print('calibration',i+1,flush=True)
    trials=[]
    for threshold in [.2,.3,.4,.5,.6,.7,.8]:
        for cut in [15,20,25,30,35]:
            accepted=[];correct=[]
            for stem,views,g,holes,gt in calibration:
                fused,reason=consensus(views,threshold)
                if reason:continue
                if checks(g,fused,holes,cut):continue
                accepted.append(stem);correct.append(true_complete(fused,gt))
            trials.append(dict(threshold=threshold,pixel_cut=cut,accepted=len(accepted),correct=sum(correct),
                               errors=len(correct)-sum(correct),accepted_stems=accepted))
    eligible=[t for t in trials if t['accepted']>=15 and t['errors']==0]
    # This is calibration evidence, not an unbiased accuracy estimate or a guarantee.
    chosen=max(eligible,key=lambda t:(t['accepted'],t['threshold'],-t['pixel_cut'])) if eligible else None
    calibration_report={'trials':trials,'chosen':chosen,'minimum_calibration_accepts':15,'strict_iou_requirement':.5,
        'uses_external_validation_or_test':False,'accuracy_is_calibration_not_independent_test':True,
        'warning':'Thresholds selected on this set; zero observed errors is not proof of zero future errors.'}
    (REPORT/'calibration.json').write_text(json.dumps(calibration_report,ensure_ascii=False,indent=2))
    decisions=[];accepted_rows=[]
    output=OUT/'pseudo_labels';output.mkdir(exist_ok=True)
    candidates=inv[inv.status=='pseudo_candidate']
    for i,(stem,r) in enumerate(candidates.iterrows()):
        views=cached(stem);threshold=chosen['threshold'] if chosen else .7
        fused,reason=consensus(views,threshold)
        if fused is not None:
            g=cv2.imread(str(ROOT/r.image),0);_,holes,_=mark_regions(load_palette(ROOT/r.source))
            reason=reason or checks(g,fused,holes,chosen['pixel_cut'] if chosen else 20)
        if not chosen:reason='calibration_insufficient'+(':'+reason if reason else '')
        status='accepted' if not reason else ('held' if 'no_detection' in reason or 'calibration_insufficient' in reason else 'review')
        proposal=fused if fused is not None else np.asarray(views[0]).reshape(-1,5)
        proposed_path=REPORT/'proposed_labels'/f'{stem}.txt';proposed_path.parent.mkdir(exist_ok=True)
        with Image.open(ROOT/r.image) as im:w,h=im.size
        put_label(proposed_path,norm(proposal[:,:4],w,h))
        if status=='accepted':
            dest=output/f'{stem}.txt';put_label(dest,norm(fused[:,:4],w,h))
            accepted_rows.append(dict(stem=stem,image=r.image,label=str(dest.relative_to(ROOT)),group=int(r.group),n_boxes=len(fused)))
        decisions.append(dict(stem=stem,status=status,reason=reason,n_boxes=len(proposal),group=int(r.group),
                              source=r.source,image=r.image,proposed_label=str(proposed_path.relative_to(ROOT))))
        if (i+1)%100==0:
            pd.DataFrame(decisions).to_csv(REPORT/'decisions_partial.csv',index=False);print('candidates',i+1,len(candidates),flush=True)
    pd.DataFrame(decisions).to_csv(REPORT/'decisions.csv',index=False)
    pd.DataFrame(accepted_rows,columns=['stem','image','label','group','n_boxes']).to_csv(REPORT/'accepted.csv',index=False)
    # A full-image decision: never train on a partially annotated unresolved image.
    summary={'candidates':len(candidates),'statuses':pd.Series([r['status'] for r in decisions]).value_counts().to_dict(),
        'accepted_boxes':sum(r['n_boxes'] for r in accepted_rows),'calibration':chosen,'official_labels_unchanged':True,
        'pseudo_labels_are_not_official_ground_truth':True,'no_detection_is_not_normal':True}
    (REPORT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
    if accepted_rows:
        for name,base in [('pseudo','official'),('geometry_pseudo','geometry')]:
            variant=OUT/'variants'/name
            if variant.exists():raise RuntimeError('Pseudo variant already exists; preserve previous result')
            for split_name in ['train','val','test']:
                for image in (OUT/'variants'/base/'images'/split_name).glob('*.png'):
                    link(image,variant/'images'/split_name/image.name)
                    link(OUT/'variants'/base/'labels'/split_name/(image.stem+'.txt'),variant/'labels'/split_name/(image.stem+'.txt'))
            for r in accepted_rows:
                link(ROOT/r['image'],variant/'images/train'/f"{r['stem']}.png")
                link(ROOT/r['label'],variant/'labels/train'/f"{r['stem']}.txt")
            (variant/'data.yaml').write_text(yaml.safe_dump({'path':'.','train':'images/train','val':'images/val','test':'images/test','names':{0:'defect'}}))


if __name__=='__main__':main()
