"""Run the approved two-model latest-YOLO study sequentially on the Mac."""
import subprocess
import sys
from xray_config import ROOT
from xray_recovery import atomic_json, run_lock


def main():
    folder=ROOT/'reports/design_review_20261004'
    folder.mkdir(exist_ok=True)
    jobs=['yolo26n','yolo11n']
    with run_lock(folder/'latest_queue'):
        for architecture in jobs:
            name=f'v2_latest20_{architecture}_guard_mps_20261004'
            args=[sys.executable,'scripts/xray_train_latest_yolo.py','--architecture',architecture,
                  '--name',name,'--resume']
            atomic_json(folder/'latest_queue.json',dict(status='running',current=name,jobs=jobs))
            print('Running',name,flush=True)
            with (folder/f'{architecture}_guard.log').open('a') as log:
                result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(folder/'latest_queue.json',dict(status='failed',current=name,code=result.returncode))
                raise SystemExit(result.returncode)
        atomic_json(folder/'latest_queue.json',dict(status='complete',jobs=jobs,test_evaluated=False))
        print('Both latest YOLO runs complete; test not evaluated',flush=True)


if __name__=='__main__':main()
