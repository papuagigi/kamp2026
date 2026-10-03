"""Sequence independent train/predict/evaluate programs, recording every command.

Uses one GPU worker. Existing unrelated outputs are preserved; failed runs stop the queue.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from xray_config import ROOT

PY=ROOT/'.venv/bin/python'
REPORT=ROOT/'reports/roadmap_20261002'
STATE=REPORT/'queue_status.json'


def state(**kwargs):
    kwargs['updated_unix']=time.time();STATE.write_text(json.dumps(kwargs,ensure_ascii=False,indent=2))
    print(json.dumps(kwargs,ensure_ascii=False),flush=True)


def command(args,name):
    log=REPORT/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[str(PY),*args];state(stage=name,status='running',command=args,log=str(log.relative_to(ROOT)))
    with log.open('a') as f:
        f.write('\nCOMMAND '+json.dumps(args)+'\n');f.flush()
        result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        state(stage=name,status='failed',exit_code=result.returncode,log=str(log.relative_to(ROOT)))
        raise RuntimeError(f'{name}: exit {result.returncode}')


def wait_external(name):
    path=ROOT/'runs'/name/'execution.json';state(stage=name,status='waiting_for_training')
    while True:
        data=json.loads(path.read_text())
        if data['status']=='complete':return
        if data['status'] in ['failed','interrupted']:raise RuntimeError(f'{name}: {data}')
        time.sleep(10)


def train(model,variant,name,batch=8,steps=None):
    meta=ROOT/'runs'/name/'execution.json'
    if meta.exists():
        if json.loads(meta.read_text())['status']=='complete':return
        raise RuntimeError(f'Preserved non-complete run: {name}')
    args=['scripts/xray_train_controlled.py','--model',model,'--variant',variant,'--name',name,
          '--epochs','20','--batch',str(batch),'--device','cpu' if model=='faster' else 'mps']
    if steps:args+=['--steps',str(steps)]
    command(args,'train_'+name)


def predict_score(name,split='val',threshold=None):
    csv=ROOT/'reports'/f'preds_{name}_{split}.csv';meta=csv.with_suffix('.run.json')
    if not csv.exists():
        execution=json.loads((ROOT/'runs'/name/'execution.json').read_text())
        device='cpu' if execution['model']=='faster' else 'mps'
        command(['scripts/xray_predict_controlled.py','--run',name,'--split',split,'--device',device],'predict_'+name+'_'+split)
    if not meta.exists():raise RuntimeError(f'Incomplete prediction output: {csv}')
    result=ROOT/'reports'/f'{csv.stem}_iou50_v1_eval.json'
    if not result.exists():
        args=['scripts/xray_eval.py',str(csv.relative_to(ROOT)),'--split',split,'--matching','iou50']
        if threshold is not None:args+=['--thr',str(threshold)]
        command(args,'score_'+name+'_'+split)
    return json.loads(result.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--through',choices=['labels','ablations','all'],default='all');args=parser.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True)
    os.environ.update(PYTORCH_ENABLE_MPS_FALLBACK='1',MPLCONFIGDIR='/private/tmp/xray-ml-mpl',
        TORCH_HOME=str(ROOT/'weights/torch'),HF_HOME=str(ROOT/'weights/hf'),YOLO_CONFIG_DIR='/private/tmp/xray-yolo')
    wait_external('v2_roadmap_rfdetr_official_oneclass')
    predict_score('v2_roadmap_rfdetr_official_oneclass')
    train('rfdetr','teacher','v2_roadmap_rfdetr_teacher',4)
    wait_external('v2_roadmap_yolo_teacher')
    if not (REPORT/'pseudo/summary.json').exists():command(['scripts/xray_pseudo_label.py','--device','mps'],'pseudo_labels')
    if args.through=='labels':
        state(stage='labels',status='complete_stage');return
    variants=['official','geometry','remove_all_refined','remove_partial_refined','geometry_removal_refined','offbar']
    if (ROOT/'data/xray_roadmap_20261002/variants/pseudo').exists():variants+=['pseudo','geometry_pseudo']
    results=[]
    for variant in variants:
        name='v2_roadmap_yolo_budget_'+variant
        train('yolo',variant,name,8,740)
        scores=predict_score(name)
        results.append({'variant':variant,'run':name,**scores})
        (REPORT/'ablation_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    # Predeclared conservative rule: primary F1 must improve; a tie keeps official data.
    base=results[0];best=max(results,key=lambda r:(r['f1'],r['recall'],-r['fp_per_image']))
    selected=best if best['f1']>base['f1']+1e-6 else base
    selection={'selected_variant':selected['variant'],'yolo_run':selected['run'],
        'rule':'validation F1 must strictly improve; tied F1 retains official-only data',
        'offbar_adopted':selected['variant']=='offbar','no_test_selection':True}
    (REPORT/'dataset_selection.json').write_text(json.dumps(selection,indent=2))
    if args.through=='ablations':
        state(stage='ablations',status='complete_stage');return
    wait_external('v2_roadmap_faster_official_cpu')
    cnn_run='v2_roadmap_faster_official_cpu';rf_run='v2_roadmap_rfdetr_official_oneclass'
    official=[{'run':name,'validation':predict_score(name)} for name in [base['run'],cnn_run,rf_run]]
    (REPORT/'official_baseline_comparison.json').write_text(json.dumps(official,ensure_ascii=False,indent=2))
    if selected['variant']!='official':
        cnn_run='v2_roadmap_faster_selected';rf_run='v2_roadmap_rfdetr_selected'
        train('faster',selected['variant'],cnn_run,4)
        train('rfdetr',selected['variant'],rf_run,4)
    # Final three-model comparison on the same selected data condition.
    final=[]
    for name in [selected['run'],cnn_run,rf_run]:
        val=predict_score(name)
        if val['thr'] is None:raise RuntimeError('No usable validation threshold; model requires diagnosis')
        test=predict_score(name,'test',val['thr'])
        final.append({'run':name,'validation':val,'test':test})
    (REPORT/'final_comparison.json').write_text(json.dumps(final,ensure_ascii=False,indent=2))
    state(stage='experiments',status='complete',report='reports/roadmap_20261002/final_comparison.json')


if __name__=='__main__':main()
