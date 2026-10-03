"""Refine synthetic negatives by erasing local dark components, not whole GT boxes."""
import json
import cv2
import numpy as np
import pandas as pd
import yaml
from xray_config import DATA,ROOT
from xray_prepare_roadmap import OUT,labels,xyxy,put_label,link


def refine(g,b,remove):
    h,w=g.shape;yy,xx=np.indices(g.shape);mask=np.zeros_like(g);parts=[]
    bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    for i in remove:
        cx,cy=b[i,1:3]*[w,h]
        cand=((bh>=5)&(((xx-cx)**2+(yy-cy)**2)<=25)).astype('uint8')
        n,cc,stats,cen=cv2.connectedComponentsWithStats(cand,8)
        ids=[j for j in range(1,n) if np.linalg.norm(cen[j]-[cx,cy])<=3.5]
        if not ids:return None,None,dict(passed=False,reason='cannot_isolate_dark_component')
        part=cv2.dilate(np.isin(cc,ids).astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        mask[part>0]=255;parts.append(part)
    keep=[i for i in range(len(b)) if i not in remove]
    for i in keep:
        x1,y1,x2,y2=np.round(xyxy(b[i:i+1],w,h)[0]).astype(int)
        if mask[max(0,y1):min(h,y2+1),max(0,x1):min(w,x2+1)].any():
            return None,None,dict(passed=False,reason='mask_overlaps_retained_label')
    result=cv2.inpaint(g,mask,3,cv2.INPAINT_NS)
    assert np.array_equal(result[mask==0],g[mask==0])
    remaining=cv2.morphologyEx(result,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    peaks=[]
    for i in remove:
        cx,cy=np.round(b[i,1:3]*[w,h]).astype(int)
        peaks.append(int(remaining[max(0,cy-2):cy+3,max(0,cx-2):cx+3].max()))
    passed=max(peaks,default=0)<=6
    return result,b[keep],dict(passed=passed,reason='' if passed else 'residual_dark_peak_over_6',
        residual_peaks=peaks,removed_indices=list(remove),mask_pixels=int((mask>0).sum()),
        method='9px blackhat, local component within 5px of GT center, 1px dilation, NS fill')


def main():
    man=pd.read_csv(DATA/'manifest.csv').fillna('');old=pd.read_csv(OUT/'augmentation.csv')
    if old.kind.str.endswith('_refined').any():raise RuntimeError('Refinement already exists')
    rows=[];audit=[]
    for r in man[man.split=='train'].itertuples():
        if '라벨누락' in r.label_issue:continue
        g=cv2.imread(str(OUT/'images/clean'/f'{r.stem}.png'),0);b=labels(DATA/'labels/train'/f'{r.stem}.txt')
        for basekind in ['remove_all','remove_partial']:
            if basekind=='remove_partial' and len(b)<2:continue
            previous=old[(old.parent==r.stem)&(old.kind==basekind)]
            remove=list(range(len(b))) if basekind=='remove_all' else (json.loads(previous.iloc[0].details)['removed_indices'] if len(previous) else [0])
            img,kept,qa=refine(g,b,remove);audit.append(dict(stem=r.stem,kind=basekind,**qa))
            if not qa['passed']:continue
            kind=basekind+'_refined';stem=r.stem+'__'+kind
            ip=OUT/'images/augment'/f'{stem}.png';lp=OUT/'labels/augment'/f'{stem}.txt'
            assert cv2.imwrite(str(ip),img);put_label(lp,kept)
            rows.append(dict(stem=stem,parent=r.stem,burst_id=r.burst_id,kind=kind,image=str(ip.relative_to(ROOT)),
                             label=str(lp.relative_to(ROOT)),n_boxes=len(kept),details=json.dumps(qa)))
    new=pd.DataFrame(rows);both=pd.concat([old,new],ignore_index=True);both.to_csv(OUT/'augmentation.csv',index=False)
    stats={}
    for name,kinds in {'remove_all_refined':['remove_all_refined'],'remove_partial_refined':['remove_partial_refined'],
                       'geometry_removal_refined':['crop','rotate_m','rotate_p','remove_all_refined','remove_partial_refined']}.items():
        v=OUT/'variants'/name
        for r in man.itertuples():
            link(OUT/'images/clean'/f'{r.stem}.png',v/'images'/r.split/f'{r.stem}.png')
            link(DATA/'labels'/r.split/f'{r.stem}.txt',v/'labels'/r.split/f'{r.stem}.txt')
        extra=both[both.kind.isin(kinds)]
        for r in extra.itertuples():
            link(ROOT/r.image,v/'images/train'/f'{r.stem}.png');link(ROOT/r.label,v/'labels/train'/f'{r.stem}.txt')
        (v/'data.yaml').write_text(yaml.safe_dump({'path':'.','train':'images/train','val':'images/val','test':'images/test','names':{0:'defect'}}))
        stats[name]={'train_images':296+len(extra),'added_images':len(extra),'added_boxes':int(extra.n_boxes.sum())}
    result={'method':'local dark-component removal; excludes whole-box initial prototypes from experiment selection',
            'counts':new.kind.value_counts().to_dict(),'variants':stats,'checks':audit,
            'reject_counts':pd.Series([r['reason'] for r in audit if not r['passed']]).value_counts().to_dict(),
            'source_data_unchanged':True,'original_prototypes_preserved':True}
    (OUT/'refined_removal_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
