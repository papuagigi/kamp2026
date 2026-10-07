"""Describe frozen predictions, learning records and complementarity; no tuning."""
import json
import numpy as np
import pandas as pd
from xray_config import ROOT
from xray_eval import load_gt, match, iou, evaluate
from xray_cascade_runtime import agreement
from xray_rescore_direct import RUNS, OUT


def main():
    OUT.mkdir(parents=True,exist_ok=True);errors=[];objects=[];photos=[];curves=[];summaries={};conditions=[];false_positives=[]
    for split in ['val','test']:
        gt,man=load_gt(split);selected={};detected={}
        for name,run in RUNS.items():
            r=ROOT/'runs'/run;f=json.loads((r/'evaluation_frozen.json').read_text());threshold=f['thresholds']['iou50']
            raw=pd.read_csv(r/f'predictions_{split}.csv');pred=raw[raw.score>=threshold].copy()
            paired,hits=match(pred,gt,iou_thr=.5,iou_only=True);detected[name]=np.isfinite(hits);selected[name]=pred
            for p in paired[paired.hit.lt(0)].itertuples():
                same=gt[gt.stem.eq(p.stem)]
                distance=np.hypot(same.cx-p.cx,same.cy-p.cy);idx=distance.idxmin();g=same.loc[idx]
                false_positives.append(dict(model=name,split=split,stem=p.stem,pred_cx=p.cx,pred_cy=p.cy,pred_w=p.w,pred_h=p.h,score=p.score,
                    nearest_gt=int(idx),nearest_gt_already_detected=bool(detected[name][idx]),distance=float(distance.loc[idx]),
                    iou=float(iou(np.array([p.cx,p.cy,p.w,p.h]),g[['cx','cy','w','h']].to_numpy(float)))))
            for idx,g in gt.iterrows():
                near=pred[pred.stem.eq(g.stem)].copy();gbox=g[['cx','cy','w','h']].to_numpy(float)
                if len(near):
                    overlaps=np.array([iou(x,gbox) for x in near[['cx','cy','w','h']].to_numpy()]);p=near.iloc[int(overlaps.argmax())]
                    ov=float(overlaps.max());dist=float(np.hypot(p.cx-g.cx,p.cy-g.cy));inside=bool(abs(p.cx-g.cx)<=p.w/2 and abs(p.cy-g.cy)<=p.h/2)
                    detail=dict(pred_cx=float(p.cx),pred_cy=float(p.cy),pred_w=float(p.w),pred_h=float(p.h),pred_score=float(p.score),best_iou=ov,center_distance=dist,gt_center_inside=inside,area_ratio=float(p.w*p.h/(g.w*g.h)))
                else:detail=dict(pred_cx=None,pred_cy=None,pred_w=None,pred_h=None,pred_score=None,best_iou=0.,center_distance=None,gt_center_inside=False,area_ratio=None)
                info=dict(model=name,split=split,gt_index=int(idx),detected=bool(detected[name][idx]),**g.to_dict(),**detail)
                objects.append(info)
                if not detected[name][idx]:
                    low=raw[raw.stem.eq(g.stem)&raw.score.lt(threshold)]
                    good=low[[iou(x,gbox)>=.5 for x in low[['cx','cy','w','h']].to_numpy()]] if len(low) else low
                    cause='accepted_nearby_box_iou_below_05' if detail['best_iou']>0 or (detail['center_distance'] is not None and detail['center_distance']<=8) else 'below_threshold' if len(good) else 'no_nearby_accepted_box'
                    errors.append(dict(**info,diagnostic=cause,best_below_threshold_iou50_score=float(good.score.max()) if len(good) else None))
            for field in ['machine','size','contrast','edge']:
                if field=='machine':cats=gt.machine
                elif field=='size':cats=pd.cut(gt['size'],[0,8,11,14,np.inf],labels=['<=8','8-11','11-14','>14'])
                elif field=='contrast':cats=pd.cut(gt.contrast,[-np.inf,35,45,55,np.inf],labels=['<=35','35-45','45-55','>55'])
                else:cats=pd.cut(gt.edge,[-np.inf,20,40,60,np.inf],labels=['<=20','20-40','40-60','>60'])
                for category in cats.dropna().unique():
                    choose=(cats==category).to_numpy();n=int(choose.sum());fn=int((choose&~detected[name]).sum())
                    conditions.append(dict(model=name,split=split,condition=field,category=str(category),n_gt=n,fn=fn,recall=1-fn/n))
        a,b=[detected[n] for n in RUNS];names=list(RUNS)
        for row in man.itertuples():
            first=selected[names[0]][selected[names[0]].stem.eq(row.stem)];second=selected[names[1]][selected[names[1]].stem.eq(row.stem)]
            agree=agreement(first,second)
            photos.append(dict(split=split,stem=row.stem,yolo_boxes=len(first),rf_boxes=len(second),**agree,
                any_alarm=bool(len(first) or len(second)),both_empty=not(len(first) or len(second)),
                initial_action='RE-INSPECTION' if len(first) or len(second) or not agree['agree'] else 'CHECK_NON_AI_CONDITIONS'))
        pp=[p for p in photos if p['split']==split]
        summaries[split]=dict(images=len(man),gt_objects=len(gt),both_detected=int((a&b).sum()),yolo_only=int((a&~b).sum()),
            rf_only=int((~a&b).sum()),both_iou_missed=int((~a&~b).sum()),either_iou_detected=int((a|b).sum()),
            both_empty_photos=sum(p['both_empty'] for p in pp),yolo_empty_photos=sum(p['yolo_boxes']==0 for p in pp),rf_empty_photos=sum(p['rf_boxes']==0 for p in pp),
            disagree_photos=sum(not p['agree'] for p in pp),real_normal_photos=0,
            note='Object coverage uses labels after inference; not a merged detector F1 or factory escaped-product rate. No true-normal false alarm rate.')
    for name,run in RUNS.items():
        r=ROOT/'runs'/run;selection=json.loads((r/'selection.json').read_text())
        if name=='YOLO11s':
            loss=pd.read_csv(r/'results.csv').set_index('epoch')
        else:
            l=pd.read_csv(r/'metrics.csv');l['epoch']=l.epoch+1
            loss=l.groupby('epoch')[['train/loss','val/loss']].last()
        for epoch in range(1,21):
            v=json.loads((r/f'validation/epoch_{epoch:03d}.json').read_text());m=v['metrics']['iou50'];ll=loss.loc[epoch]
            if name=='YOLO11s':
                train=float(sum(ll[k] for k in ['train/box_loss','train/cls_loss','train/dfl_loss']));val=float(sum(ll[k] for k in ['val/box_loss','val/cls_loss','val/dfl_loss']))
            else:train=float(ll['train/loss']);val=float(ll['val/loss'])
            curves.append(dict(model=name,epoch=epoch,selected=epoch==selection['best_epoch'],validation_f1=m['f1'],validation_ap50=m['ap'],validation_threshold=m['thr'],train_loss=train,val_loss=val,device=v['device']))
    for filename,data in [('objects.csv',objects),('fn_diagnostics.csv',errors),('fp_diagnostics.csv',false_positives),('photo_decisions.csv',photos),('learning_curves.csv',curves),('conditions.csv',conditions)]:
        pd.DataFrame(data).to_csv(OUT/filename,index=False)
    summary=dict(splits=summaries,fn_diagnostics=pd.DataFrame(errors).groupby(['model','split','diagnostic']).size().rename('count').reset_index().to_dict('records'),
        scope='Saved predictions and native epoch losses; no new training, threshold selection or test optimization.',
        loss_note='YOLO sum of its three logged components; RF native total. Compare each model across epochs only; not loss values between models.',
        human_label_accuracy=None,visual_review_record='visual_review/decisions.json; separate qualitative AI review, not rescoring')
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def device_comparison():
    """Re-score fresh benchmark CSVs at frozen thresholds; never retune."""
    if not all((OUT/f'{n}_live_val.csv').exists() for n in RUNS):return
    rows=[];live_hits={};live_metrics={};changed=[]
    gt,_=load_gt('val')
    for name,run in RUNS.items():
        r=ROOT/'runs'/run;threshold=json.loads((r/'evaluation_frozen.json').read_text())['thresholds']['iou50']
        raw=pd.read_csv(r/'predictions_val.csv');live=pd.read_csv(OUT/f'{name}_live_val.csv')
        _,saved_hit=match(raw[raw.score>=threshold],gt,iou_thr=.5,iou_only=True)
        metrics,_,lg=evaluate(OUT/f'{name}_live_val.csv','val',thr=threshold,matching='iou50')
        live_hits[name]=lg.detected.to_numpy();live_metrics[name]=metrics
        for idx in np.flatnonzero(np.isfinite(saved_hit)!=live_hits[name]):
            g=gt.iloc[idx];entry=dict(model=name,gt_index=int(idx),stem=g.stem,saved_detected=bool(np.isfinite(saved_hit[idx])),mps_detected=bool(live_hits[name][idx]),threshold=threshold)
            for tag,df in [('saved',raw),('live',live)]:
                same=df[df.stem.eq(g.stem)].copy();overlap=[iou(x,g[['cx','cy','w','h']].to_numpy(float)) for x in same[['cx','cy','w','h']].to_numpy()]
                near=same.iloc[int(np.argmax(overlap))];entry[tag+'_score']=float(near.score);entry[tag+'_iou']=float(max(overlap))
            changed.append(entry)
    a,b=[live_hits[n] for n in RUNS]
    result=dict(changed_gt=changed,live_device='mps',live_metrics=live_metrics,
        live_pair=dict(both=int((a&b).sum()),yolo_only=int((a&~b).sum()),rf_only=int((~a&b).sum()),both_iou_missed=int((~a&~b).sum())),threshold_changed=False)
    (OUT/'device_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main();device_comparison()
