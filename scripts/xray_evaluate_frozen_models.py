"""Run fixed-checkpoint test inference and score at validation-chosen thresholds."""
import hashlib,json,os,subprocess
from pathlib import Path
from xray_config import ROOT,DATA
from xray_eval import evaluate,MATCH_POLICIES

RUNS=['v2_roadmap_yolo_official','v2_roadmap_faster_official_cpu','v2_roadmap_rfdetr_official_oneclass']

def main():
    report=ROOT/'reports/roadmap_20261002'
    policy=json.loads((report/'evaluation_policy_20261002.json').read_text())
    results=[]
    for name in RUNS:
        execution=json.loads((ROOT/'runs'/name/'execution.json').read_text())
        assert execution['status']=='complete'
        csv=ROOT/'reports'/f'preds_{name}_test.csv'
        if not csv.exists():
            device='mps' if execution['model']=='rfdetr' else 'cpu'
            subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/xray_predict_controlled.py','--run',name,'--split','test','--device',device],cwd=ROOT,check=True)
        metadata=json.loads(csv.with_suffix('.run.json').read_text())
        assert metadata['checkpoint_sha256']==hashlib.sha256((ROOT/execution['checkpoint']).read_bytes()).hexdigest()
        assert metadata['images']==97
        results_for_run={'run':name,'execution':execution,'inference':metadata,'metrics':{}}
        for mode,(_,_,version) in MATCH_POLICIES.items():
            val_path=ROOT/'reports'/f'preds_{name}_val_{version}_eval.json';val=json.loads(val_path.read_text());assert val['thr'] is not None
            test,_,_=evaluate(csv,'test',thr=val['thr'],matching=mode)
            test['threshold_provenance']=str(val_path.relative_to(ROOT));test['threshold_policy']='fixed_from_validation_no_test_optimization'
            out=csv.with_name(csv.stem+'_'+version+'_eval.json');out.write_text(json.dumps(test,ensure_ascii=False,indent=2))
            results_for_run['metrics'][mode]={'validation':val,'test':test}
        results.append(results_for_run)
        (report/'official_test_comparison.json').write_text(json.dumps({'policy':policy,'models':results,'model_selection_uses_test':False},ensure_ascii=False,indent=2))
        print(json.dumps({'run':name,**{mode:{k:r['test'][k] for k in ['thr','tp','fp','fn','f1','ap']} for mode,r in results_for_run['metrics'].items()}},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
