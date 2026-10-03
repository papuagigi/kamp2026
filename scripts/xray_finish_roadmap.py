"""Build review deliverables once the running experiment queue finishes successfully."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from xray_config import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--pdf-libs',required=True);a=p.parse_args()
    report=ROOT/'reports/roadmap_20261002';state=report/'delivery_status.json'
    state.write_text(json.dumps({'status':'waiting_for_experiments'}))
    while not (report/'final_comparison.json').exists():
        if (report/'queue_status.json').exists():
            q=json.loads((report/'queue_status.json').read_text())
            if q['status']=='failed':raise RuntimeError(q)
        time.sleep(10)
    commands=[['scripts/xray_benchmark_models.py','--device','cpu'],['scripts/xray_validate_roadmap.py'],['scripts/xray_review_gallery.py'],
              ['scripts/xray_results_report.py','--pdf-libs',a.pdf_libs],
              ['scripts/xray_product_outputs.py'],['scripts/xray_package_reproduction.py']]
    try:
        for args in commands:
            stage=Path(args[0]).stem;state.write_text(json.dumps({'status':'running','stage':stage}))
            with (report/'logs'/f'{stage}.log').open('a') as f:
                subprocess.run([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        state.write_text(json.dumps({'status':'generated','visual_verification':'pending'}))
    except Exception as exc:
        state.write_text(json.dumps({'status':'failed','stage':stage,'error':str(exc)}));raise

if __name__=='__main__':main()
