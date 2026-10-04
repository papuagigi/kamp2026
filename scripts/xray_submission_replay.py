"""Verify exported inputs and rescore saved test CSVs with the canonical scorer.

Uses the currently selected Python environment; does not retrain models.
"""
import hashlib
import json
from pathlib import Path
from xray_config import ROOT
from xray_eval import evaluate


def main():
    manifest=json.loads((ROOT/'PACKAGE_SHA256.json').read_text())
    for name, expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError(name)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('Checksum mismatch: '+name)
    policy=json.loads((ROOT/'reproduction_policy.json').read_text())
    rows=[]
    for row in policy['models']:
        m,_,_=evaluate(ROOT/row['predictions'],'test',thr=row['threshold'],matching='iou50')
        for key,value in row['expected'].items():
            if abs(m[key]-value)>1e-10:raise ValueError(f"Metric mismatch: {row['name']} {key}")
        rows.append(dict(name=row['name'],**{k:m[k] for k in row['expected']}))
    result=dict(status='passed',checked_files=len(manifest['files']),models=rows,
                scope='exported files checksum and saved test prediction rescore',
                full_retraining=False,fresh_environment=False,new_inference=False,
                original_split_preserved=True)
    (ROOT/'replay_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
