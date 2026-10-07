"""Evaluate one completed local run without polling through an AI agent."""
import argparse
import json
import subprocess
import sys
import time
from xray_config import ROOT
from xray_recovery import atomic_json, run_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run
    assert run.parent == ROOT / 'runs'
    with run_lock(run / 'evaluation_waiter'):
        state = run / 'evaluation_waiter.json'
        deadline = time.monotonic() + 16 * 3600
        atomic_json(state, dict(status='waiting', run=args.run, started_unix=time.time()))
        while time.monotonic() < deadline:
            try:
                meta = json.loads((run / 'execution.json').read_text())
            except (OSError, ValueError):
                time.sleep(30)
                continue
            if meta['status'] == 'complete':
                with (run / 'evaluation.log').open('a') as log:
                    rc = subprocess.run([sys.executable, 'scripts/xray_evaluate_direct.py',
                        '--run', args.run, '--device', args.device], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT).returncode
                atomic_json(state, dict(status='complete' if rc == 0 else 'failed',
                    returncode=rc, updated_unix=time.time()))
                raise SystemExit(rc)
            if meta['status'] in {'failed', 'interrupted', 'paused'}:
                atomic_json(state, dict(status='training_stopped',
                    training_status=meta['status'], updated_unix=time.time()))
                return
            time.sleep(30)
        atomic_json(state, dict(status='wait_timeout', updated_unix=time.time()))


if __name__ == '__main__':
    main()
