"""Prepare anonymous visual-review evidence; never infer visual decisions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from xray_config import DATA, ROOT
from xray_selected_common import RUNS, REPORT

OUT = REPORT / 'visual_review'
FONT = ImageFont.load_default(size=14)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    gpath = REPORT/'diagnostics/ground_truth_detail.csv'
    ppath = REPORT/'diagnostics/selected_predictions.csv'
    gt = pd.read_csv(gpath).drop_duplicates(['split', 'gt_index'])
    pred = pd.read_csv(ppath)
    rng = np.random.default_rng(0)
    stems = sorted(gt.loc[gt.split == 'val', 'stem'].unique())
    pilot = set(rng.choice(stems, 12, replace=False))
    selected = gt[(gt.split == 'test') | ((gt.split == 'val') & gt.stem.isin(pilot))]
    records = []
    for split in ['val', 'test']:
        subset = selected[selected.split == split].sort_values('gt_index')
        for j, g in enumerate(subset.itertuples(), 1):
            cid = ('V' if split == 'val' else 'T') + f'{j:03}'
            source = DATA/'images'/split/f'{g.stem}.png'
            with Image.open(source) as im:
                im = im.convert('RGB')
            group = gt[(gt.split == split) & (gt.stem == g.stem)]
            boxes = pred[(pred.split == split) & (pred.stem == g.stem)].copy()
            # Nearest reference routes a box to a display row only. The AI must
            # inspect every box; this assignment is NOT a TP/FN decision.
            near = []
            for p in boxes.itertuples():
                k = ((group.cx-p.cx)**2 + (group.cy-p.cy)**2).idxmin()
                near.append(int(group.loc[k, 'gt_index']))
            boxes = boxes.assign(display_gt=near)
            boxes = boxes[boxes.display_gt == g.gt_index]
            half = max(16., g.w, g.h)
            for p in boxes.itertuples():
                half = max(half, abs(p.cx-g.cx)+p.w/2+5, abs(p.cy-g.cy)+p.h/2+5)
            side = int(np.ceil(2*half))
            x0, y0 = int(g.cx-side/2), int(g.cy-side/2)
            crop = im.crop((x0, y0, x0+side, y0+side))
            panel_size = 176
            base = crop.resize((panel_size, panel_size), Image.Resampling.NEAREST)
            row = Image.new('RGB', (5*panel_size, panel_size+35), 'white')
            row.paste(base, (0, 28));draw = ImageDraw.Draw(row)
            draw.text((3, 4), f'{cid} clean / target center', fill='black', font=FONT)
            panels = {}
            for slot, run in enumerate(rng.permutation(RUNS), 1):
                code = 'ABCD'[slot-1]
                panel = base.copy(); pdra = ImageDraw.Draw(panel)
                prs = []
                for number, p in enumerate(boxes[boxes.run == run].sort_values('pred_index').itertuples(), 1):
                    token = f'p{number}'
                    coords = [(p.cx-p.w/2-x0)*panel_size/side,
                              (p.cy-p.h/2-y0)*panel_size/side,
                              (p.cx+p.w/2-x0)*panel_size/side,
                              (p.cy+p.h/2-y0)*panel_size/side]
                    pdra.rectangle(coords, outline='#f58b00', width=2)
                    pdra.text((max(0,coords[0]), max(0,coords[1]-16)),token,fill='#c95000',font=FONT)
                    prs.append(dict(token=token, pred_index=int(p.pred_index),
                                    box=[p.cx,p.cy,p.w,p.h], score=p.score))
                row.paste(panel, (slot*panel_size,28))
                draw.text((slot*panel_size+5,4), f'{code}: '+('/'.join(x['token'] for x in prs) or 'EMPTY'), fill='black',font=FONT)
                panels[code] = dict(run=str(run), predictions=prs)
            file = OUT/'rows'/f'{cid}.png';file.parent.mkdir(exist_ok=True);row.save(file)
            records.append(dict(case=cid,split=split,stem=g.stem,gt_index=int(g.gt_index),
                reference_box=[g.cx,g.cy,g.w,g.h],source=str(source.relative_to(ROOT)),
                source_sha256=sha(source),crop_xyxy=[x0,y0,x0+side,y0+side],
                panels=panels,figure=str(file.relative_to(ROOT)),figure_sha256=sha(file)))
        split_records=[x for x in records if x['split']==split]
        for offset in range(0,len(split_records),6):
            batch=split_records[offset:offset+6]
            sheet=Image.new('RGB',(880,211*len(batch)),'white')
            for n,item in enumerate(batch):
                with Image.open(ROOT/item['figure']) as row:sheet.paste(row,(0,n*211))
            sheet.save(OUT/f'{split}_{offset//6+1:02}.png')
    policy = dict(version='ai_visual_review_v1', seed=0,
        purpose='Exploratory AI-assisted review of visible foreign-body indication, not factory deployment validation.',
        reviewer='Codex in this conversation; model names, score and IoU omitted from panels; prior results already seen.',
        positive='The compact dark core of the labeled target is visibly inside the box, which indicates this local target rather than the whole product. Not a pixel-perfect contour judgment.',
        negative='The box visibly fails to contain the target core or marks a different background feature.',
        uncertain='Core or boundary is ambiguous, target is clipped, box is excessively broad, or an unlabelled candidate appears. Never silently turn uncertainty into success.',
        reference='Official TXT provides target identity. A nearest-reference assignment is only a display index, never an automatic visual match.',
        duplicate='One prediction and one official target can contribute at most one TP; extra duplicates are FP.',
        unresolved='Publish unresolved cases and score bounds; do not drop them from denominators.',
        scope='12 validation images for a reviewer preflight and all 97 previously viewed test images. No new threshold selection.',
        limitations=['Single AI reviewer; no independent expert validation.',
            'A preflight on official targets is not proof of AI judgment accuracy on new X-rays.',
            'No real normal products; no normal-product specificity or manufacturing throughput claim.'],
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [gpath,ppath,REPORT/'frozen_selection.json']})
    for name,payload in [('policy.json',policy),('case_manifest.json',records)]:
        path=OUT/name
        if path.exists():assert json.loads(path.read_text())==payload, 'Existing protocol changed'
        else:path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'validation_images':len(pilot),'validation_targets':sum(r['split']=='val' for r in records),
        'test_images':int(selected[selected.split=='test'].stem.nunique()),'test_targets':sum(r['split']=='test' for r in records),
        'displayed_predictions':{s:sum(len(p['predictions']) for r in records if r['split']==s for p in r['panels'].values()) for s in ['val','test']}},indent=2))


if __name__ == '__main__':main()
