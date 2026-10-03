"""Read-only audit of the supplied 2026-10-02 X-ray dataset; writes evidence only."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
import unicodedata
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from xray_prepare_v2 import label_issue

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/provided_preprocessing_audit_20261002'
DS = ROOT / '제조AI데이터셋/4. X-ray 검사장비 AI 데이터셋/dataset'
RAW = DS / 'test1/yolov3/X선이물검출기(06.23_09.22)'
LAB = DS / '라벨링 6종 세트/labels'
FOLDER = next(p for p in (ROOT/'preprocessed_data').iterdir() if p.is_dir() and unicodedata.normalize('NFC',p.name)=='전처리된데이터셋') / 'X-ray_데이터셋'
PAT = re.compile(r'^(h[123]_\d{3}_\d{8}_\d{6}\(\d+\))(.*)$')
SPLITS = {'학습':'train', '검증':'val', '테스트':'test'}


def labels(path):
    a = np.array([list(map(float,l.split())) for l in path.read_text().splitlines() if l.strip()], dtype=float)
    return a.reshape(-1,5)


def same_labels(a,b):
    return a.shape == b.shape and np.allclose(a,b,atol=1.1e-6,rtol=0)


def rect(b,w,h):
    _,cx,cy,bw,bh=b
    return max(0,int(np.floor((cx-bw/2)*w))),max(0,int(np.floor((cy-bh/2)*h))),min(w,int(np.ceil((cx+bw/2)*w))),min(h,int(np.ceil((cy+bh/2)*h)))


def roi(a,b):
    h,w=a.shape[:2]; x1,y1,x2,y2=rect(b,w,h)
    return a[y1:y2,x1:x2]


def product_mask(gray):
    edges=np.concatenate([gray[:5].ravel(),gray[-5:].ravel(),gray[:,:5].ravel(),gray[:,-5:].ravel()])
    m=(gray.astype(float)<np.median(edges)-25).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
    return (lab==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.uint8) if n>1 else m


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else str(o)))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    issues=[]; files=[]; augrows=[]; boxrows=[]; baserows=[]; erase=[]; colorrows=[]; equipment_disagreements=[]
    raw=[]; seen=set()
    for p in sorted(RAW.rglob('*.bmp')):
        md5=hashlib.md5(p.read_bytes()).hexdigest()
        if (p.stem,md5) in seen: continue
        seen.add((p.stem,md5))
        t=re.match(r'\d{3}_(\d{8})_(\d{6})\(\d+\)',p.stem)
        raw.append({'stem':p.stem,'md5':md5,'hogi':int(p.relative_to(RAW).parts[0][0]),'dt':datetime.strptime(t[1]+t[2],'%Y%m%d%H%M%S'),'path':p})
    raw.sort(key=lambda r:(r['hogi'],r['dt'],r['stem']))
    group=0; prev=None
    for r in raw:
        if prev is None or r['hogi']!=prev['hogi'] or (r['dt']-prev['dt']).total_seconds()>60: group+=1
        r['group']=group;prev=r
    rawmap=defaultdict(list)
    for r in raw: rawmap['h'+str(r['hogi'])+'_'+r['stem']].append(r)
    report=pd.read_csv(FOLDER/'split_report.csv')
    rep={r['name']:r for r in report.to_dict('records')}
    official={p.stem:p for p in LAB.glob('*.txt')}
    cores={}
    entries={}; parents={}; hashgroups=defaultdict(list); rgb_total=0
    for kor,split in SPLITS.items():
        imgs={p.stem:p for p in (FOLDER/kor/'images').glob('*.png')}
        labs={p.stem:p for p in (FOLDER/kor/'labels').glob('*.txt')}
        if imgs.keys()!=labs.keys(): issues.append({'type':'pairing','split':split,'image_only':sorted(imgs.keys()-labs.keys()),'label_only':sorted(labs.keys()-imgs.keys())})
        for name,p in sorted(imgs.items()):
            a=np.asarray(Image.open(p).convert('RGB')); h,w=a.shape[:2]
            gray=a[:,:,0]; color=int((a.max(2)!=a.min(2)).sum()); rgb_total+=color
            b=labels(labs[name]); m=PAT.match(name)
            if not m: raise ValueError(name)
            parent,suffix=m.groups()
            category=('original' if not suffix else 'syn_random' if suffix.startswith('_syn_random') else 'syn_edge_rot' if suffix.startswith('_syn_edge_rot') else 'clean' if suffix=='_clean' else 'partial' if suffix.startswith('_partial_') else 'unknown')
            digest=hashlib.sha256(gray.tobytes()+str(gray.shape).encode()).hexdigest()
            hashgroups[digest].append((split,name))
            bad=[]
            if len(b):
                if not np.isfinite(b).all(): bad.append('nonfinite')
                if (b[:,0]!=0).any(): bad.append('class')
                if (b[:,3:]<=0).any(): bad.append('nonpositive_size')
                if (b[:,1:3]-b[:,3:]/2 < -1e-6).any() or (b[:,1:3]+b[:,3:]/2>1+1e-6).any(): bad.append('out_of_bounds')
                if len(np.unique(b,axis=0))<len(b): bad.append('duplicate_box')
            if bad: issues.append({'type':'labels','name':name,'issues':bad})
            if color: issues.append({'type':'colored_output','name':name,'colored_pixels':color})
            record={'split':split,'name':name,'parent':parent,'category':category,'suffix':suffix,'hogi':int(parent[1]),'width':w,'height':h,'n_boxes':len(b),'colored_pixels':color,'pixel_sha256':digest}
            files.append(record); entries[name]={'record':record,'path':p,'boxes':b}
            if category=='original': parents[name]={'gray':gray.copy(),'boxes':b,'split':split,'path':p}
    print('Decoded images and labels:',len(files),flush=True)
    for name,e in parents.items():
        b=e['boxes']; g=e['gray']; h,w=g.shape
        sources=rawmap[name]
        if len(sources)!=1: raise ValueError(('ambiguous source',name,sources))
        rr=sources[0]; original=Image.open(rr['path']); idx=np.asarray(original); rgb=np.asarray(original.convert('RGB'))
        if idx.shape!=g.shape: raise ValueError(('shape',name))
        ob=labels(official[rr['stem']]); label_equal=same_labels(b,ob)
        equipment_issue=label_issue(idx,ob,w,h)
        if equipment_issue:
            equipment_disagreements.append({'name':name,'split':e['split'],'issue':equipment_issue})
        if not label_equal: issues.append({'type':'original_label_changed','name':name})
        mark=idx>=244; grow=cv2.dilate(mark.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        delta=np.abs(g.astype(float)-idx.astype(float));gtmask=np.zeros(g.shape,np.uint8)
        ordered=b[np.argsort(b[:,2])]
        yy,xx=np.mgrid[:h,:w]
        blurred=cv2.GaussianBlur(idx.astype(np.float32),(0,0),0.8)
        for k,bb in enumerate(ordered):
            x1,y1,x2,y2=rect(bb,w,h);gtmask[y1:y2,x1:x2]=1
            near=(np.hypot(xx-bb[1]*w,yy-bb[2]*h)<=4)&(~mark)
            y,x=np.unravel_index(np.argmin(np.where(near,blurred,np.inf)),idx.shape)
            cores[(rr['stem'],k)]=(x,y)
            baserows.append({'name':name,'split':e['split'],'pos_rank':k,'box_color_px':int(roi(mark,bb).sum()),'box_changed_gray_px':int(roi((delta>0)&(~mark),bb).sum()),'core_x':x,'core_y':y,'core_raw_gray':int(idx[y,x]),'core_shared_gray':int(g[y,x]),'core_abs_change':int(delta[y,x]),'core_radius2_gray_changed_px':int(((np.hypot(xx-x,yy-y)<=2)&(~mark)&(delta>0)).sum()),'core_radius3_gray_changed_px':int(((np.hypot(xx-x,yy-y)<=3)&(~mark)&(delta>0)).sum())})
        d={'name':name,'split':e['split'],'hogi':rr['hogi'],'raw_stem':rr['stem'],'raw_path':rr['path'].relative_to(ROOT).as_posix(),'raw_md5':rr['md5'],'recomputed_group':rr['group'],'declared_group':int(rep[name]['group']),'dimensions_unchanged':True,'original_labels_equal':label_equal,'raw_mark_px':int(mark.sum()),'raw_gray_px_changed_outside_mark_and_1px':int(((delta>0)&(~grow)).sum()),'gt_intersects_mark':bool((gtmask&mark).any()),'gt_intersects_grown_mark':bool((gtmask&grow).any()),'gt_gray_px_changed':int(((delta>0)&(~mark)&(gtmask>0)).sum()),'source_mapping_agrees':str(rep[name]['source']).endswith(rr['path'].relative_to(RAW).as_posix())}
        e['source']=rr; e['product']=product_mask(g)
        e['product_dist']=cv2.distanceTransform(e['product'],cv2.DIST_L2,5)
        colorrows.append(d)
    base_map=pd.DataFrame(colorrows)
    base_map.to_csv(OUT/'original_mapping.csv',index=False)
    pd.DataFrame(equipment_disagreements).to_csv(OUT/'equipment_label_disagreements.csv',index=False)
    pd.DataFrame(baserows).to_csv(OUT/'original_box_changes.csv',index=False)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    for name,e in entries.items():
        r=e['record']
        if r['category']=='original': continue
        par=parents.get(r['parent'])
        if par is None or par['split']!='train' or r['split']!='train':
            issues.append({'type':'augmentation_parent_leak','name':name});continue
        g=np.asarray(Image.open(e['path']).convert('L')); pg=par['gray']; pb=par['boxes']; b=e['boxes'];h,w=g.shape
        if g.shape!=pg.shape: issues.append({'type':'aug_dimensions','name':name});continue
        delta=pg.astype(float)-g.astype(float); record={'name':name,'parent':r['parent'],'category':r['category'],'n_boxes':len(b),'changed_px':int((delta!=0).sum()),'darker_px':int((delta>0).sum()),'lighter_px':int((delta<0).sum())}
        if r['category'].startswith('syn_'):
            kept=[any(np.allclose(bb,qq,atol=1.1e-6,rtol=0) for qq in b) for bb in pb]
            extra=[bb for bb in b if not any(np.allclose(bb,qq,atol=1.1e-6,rtol=0) for qq in pb)]
            record.update({'old_labels_retained':all(kept),'new_boxes':len(extra)})
            if not all(kept) or not 1<=len(extra)<=3: issues.append({'type':'synthetic_labels','name':name,'new_boxes':len(extra),'old_retained':all(kept)})
            for k,bb in enumerate(extra):
                patch=roi(delta,bb); center=bb[1:3]*[w,h];old_centers=pb[:,1:3]*[w,h]
                boxrows.append({'name':name,'category':r['category'],'new_box_index':k,'cx':center[0],'cy':center[1],'w':bb[3]*w,'h':bb[4]*h,'darker_px':int((patch>0).sum()),'max_darkening':float(patch.max()),'mean_darkening':float(patch.mean()),'product_fraction_proxy':float(roi(par['product'],bb).mean()),'nearest_old_center_px':float(np.linalg.norm(old_centers-center,axis=1).min())})
                boxrows[-1]['center_to_product_edge_proxy_px']=float(par['product_dist'][min(h-1,int(center[1])),min(w-1,int(center[0]))])
            record['old_box_changed_px']=sum(int((roi(delta,bb)!=0).sum()) for bb in pb)
        elif r['category'] in ('clean','partial'):
            ordered=pb[np.argsort(pb[:,2])];pattern='x'*len(pb) if r['category']=='clean' else r['suffix'].split('_')[-1]
            expected=np.array([bb for bb,o in zip(ordered,pattern) if o=='o']).reshape(-1,5)
            check=same_labels(b[np.argsort(b[:,2])],expected)
            record['pattern_labels_match']=check
            if not check: issues.append({'type':'partial_labels','name':name})
            response=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,kernel); oldresponse=cv2.morphologyEx(pg,cv2.MORPH_BLACKHAT,kernel)
            stem=r['parent'][3:]
            for k,(bb,o) in enumerate(zip(ordered,pattern)):
                x,y=cores[(stem,k)]; core_slice=(slice(max(0,y-2),min(h,y+3)),slice(max(0,x-2),min(w,x+3)))
                erase.append({'name':name,'parent':r['parent'],'pos_rank':k,'category':r['category'],'retained':o=='o','box_changed_px':int((roi(delta,bb)!=0).sum()),'core_changed_px':int((delta[core_slice]!=0).sum()),'old_blackhat':int(oldresponse[core_slice].max()),'new_blackhat':int(response[core_slice].max()),'core_old_gray':int(pg[y,x]),'core_new_gray':int(g[y,x])})
        augrows.append(record)
    df=pd.DataFrame(files); au=pd.DataFrame(augrows); nb=pd.DataFrame(boxrows); er=pd.DataFrame(erase);bm=base_map;bc=pd.DataFrame(baserows)
    for nm,dd in [('files.csv',df),('augmentations.csv',au),('synthetic_boxes.csv',nb),('erased_boxes.csv',er)]: dd.to_csv(OUT/nm,index=False)
    declared_leak=bm.groupby('declared_group').split.nunique();independent_leak=bm.groupby('recomputed_group').split.nunique()
    source_hash_leak=bm.groupby('raw_md5').split.nunique()
    dup=[v for v in hashgroups.values() if len(v)>1];crossdup=[v for v in dup if len(set(x[0] for x in v))>1]
    decoys=[]
    for split in ['val','test']:
        data=json.loads((FOLDER/f'decoys_{split}.json').read_text())
        for fname,rects in data.items():
            par=parents[Path(fname).stem];h,w=par['gray'].shape
            for x,y,bw,bh in rects:
                inside=0<=x<x+bw<=w and 0<=y<y+bh<=h
                overlap=[]
                for bb in par['boxes']:
                    x1,y1,x2,y2=rect(bb,w,h);overlap.append(max(0,min(x+bw,x2)-max(x,x1))*max(0,min(y+bh,y2)-max(y,y1)))
                decoys.append({'name':Path(fname).stem,'split':split,'x':x,'y':y,'w':bw,'h':bh,'in_bounds':inside,'overlap_gt_px':max(overlap,default=0)})
    de=pd.DataFrame(decoys);de.to_csv(OUT/'decoys.csv',index=False)
    missing=set(official)-{x[3:] for x in parents};counts=df.groupby(['split','category']).agg(images=('name','size'),boxes=('n_boxes','sum')).reset_index().to_dict('records')
    removed=er[~er.retained];retained=er[er.retained]
    parentcount=df[df.category!='original'].groupby(['parent','category']).size().unstack(fill_value=0)
    summary={'counts':counts,'all_images':len(df),'all_boxes':int(df.n_boxes.sum()),'all_unique_raw':len(raw),'all_raw_groups':group,'provided_originals':len(parents),'original_label_files':len(official),'official_labels_not_in_package':sorted(missing),'label_and_pair_errors':issues,'rgb_colored_pixels_all_outputs':rgb_total,'base_dimensions_unchanged':bool(bm.dimensions_unchanged.all()),'original_labels_all_equal':bool(bm.original_labels_equal.all()),'source_paths_all_agree':bool(bm.source_mapping_agrees.all()),'declared_groups_crossing_splits':int((declared_leak>1).sum()),'independent_groups_crossing_splits':int((independent_leak>1).sum()),'raw_md5_crossing_splits':int((source_hash_leak>1).sum()),'identical_pixel_groups':dup,'identical_pixel_groups_crossing_splits':crossdup,'declared_metadata_counts':report.split.value_counts().to_dict(),'metadata_names_absent_from_delivered_originals':len(set(report.name)-set(parents)),'per_parent_augmentation_counts':{k:parentcount[k].value_counts().to_dict() for k in parentcount},'partial_patterns':df[df.category=='partial'].suffix.value_counts().to_dict(),'base_boxes_core_changed':int((bc.core_abs_change>0).sum()),'base_boxes_core_change_gt_10':int((bc.core_abs_change>10).sum()),'base_box_max_core_change':int(bc.core_abs_change.max()),'base_images_gt_intersects_mark':int(bm.gt_intersects_mark.sum()),'base_images_gt_intersects_mark_plus_1px':int(bm.gt_intersects_grown_mark.sum()),'base_images_any_gt_gray_change':int((bm.gt_gray_px_changed>0).sum()),'new_synthetic_boxes':len(nb),'new_boxes_with_no_darkened_pixels':int((nb.darker_px==0).sum()),'new_boxes_max_darkening_le_3':int((nb.max_darkening<=3).sum()),'new_boxes_not_fully_in_proxy_product':int((nb.product_fraction_proxy<1).sum()),'new_boxes_min_distance_to_old_centers':float(nb.nearest_old_center_px.min()),'synthetic_images_old_box_pixels_changed':int((au[au.category.str.startswith('syn_')].old_box_changed_px>0).sum()),'erased_box_count':len(removed),'removed_box_blackhat_before_median':float(removed.old_blackhat.median()),'removed_box_blackhat_after_median':float(removed.new_blackhat.median()),'removed_box_blackhat_ge_20':int((removed.new_blackhat>=20).sum()),'erased_boxes_no_pixel_change':int((removed.box_changed_px==0).sum()),'retained_boxes_pixels_changed':int((retained.box_changed_px>0).sum()),'decoy_rectangles':len(de),'decoy_out_of_bounds':int((~de.in_bounds).sum()),'decoy_rectangles_overlapping_gt':int((de.overlap_gt_px>0).sum()),'unverified':['generation code and patch donor provenance absent','decoy false-positive-rate claims require predictions and scoring','product proxy is an audit heuristic, not the producer mask','synthetic clean images do not establish performance on genuine normal production images']}
    dump('audit_summary.json',summary)
    v2=pd.read_csv(ROOT/'data/xray_v2/manifest.csv')
    v2['name']='h'+v2.machine.str[0]+'_'+v2.stem
    cmp=bm[['name','split']].merge(v2[['name','split']],on='name',suffixes=('_shared','_v2'))
    cmp.to_csv(OUT/'split_vs_v2.csv',index=False)
    risk=cmp[(cmp.split_shared=='train')&(cmp.split_v2!='train')]
    risky_aug=df[(df.category!='original')&df.parent.isin(risk.name)]
    extras={'core_radius2_changed_boxes':int((bc.core_radius2_gray_changed_px>0).sum()),'core_radius3_changed_boxes':int((bc.core_radius3_gray_changed_px>0).sum()),'split_v2_mixing_risky_originals':len(risk),'split_v2_mixing_risky_augmented_images':len(risky_aug),'by_machine_split':df.groupby(['split','category','hogi']).agg(images=('name','size'),boxes=('n_boxes','sum')).reset_index().to_dict('records'),'labeled_groups_per_split':bm.groupby('split').recomputed_group.nunique().to_dict(),'no_real_normal_val_test':bool((df[df.split!='train'].n_boxes>0).all())}
    dump('additional_checks.json',extras)
    nb.groupby('category').agg(boxes=('name','size'),median_center_edge_distance_px=('center_to_product_edge_proxy_px','median'),center_within_15px_of_edge=('center_to_product_edge_proxy_px',lambda x:int((x<=15).sum()))).to_csv(OUT/'synthetic_edge_distribution.csv')
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else str(o)))
    print(json.dumps(extras,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
