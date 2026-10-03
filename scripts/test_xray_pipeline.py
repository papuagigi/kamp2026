"""학습을 실행하지 않고 셸 파이프라인의 데이터·임계값 전달을 검사한다."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from xray_config import ROOT
from xray_check_dataset import check_dataset


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, version, mode="all", threshold=0.42):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        shutil.copy(ROOT / "run.sh", root / "run.sh")
        (root / ".venv/bin").mkdir(parents=True)
        stub = root / "stub.py"
        stub.write_text('''import json, os, sys
from pathlib import Path
if sys.argv[1] == "-":
    sys.argv = sys.argv[1:]
    exec(compile(sys.stdin.read(), "<threshold_reader>", "exec"))
else:
    args = sys.argv[1:]
    with Path("calls.jsonl").open("a") as f:
        f.write(json.dumps({"args": args, "data": os.environ["XRAY_DATA"]}) + "\\n")
    if args[0] == "scripts/xray_eval.py" and args[args.index("--split") + 1] == "val":
        Path("reports").mkdir(exist_ok=True)
        output = Path("reports") / (Path(args[1]).stem + "_custom_ap_v2_eval.json")
        output.write_text(json.dumps({"thr": json.loads(os.environ["STUB_THRESHOLD"])}))
''')
        fake_python = root / ".venv/bin/python"
        fake_python.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(stub)) + ' "$@"\n')
        fake_python.chmod(0o755)
        env = os.environ.copy()
        for key in ["XRAY_DATA", "NAME", "MODEL", "IMGSZ", "EPOCHS", "DEVICE", "RUN_DOG"]:
            env.pop(key, None)
        env.update(RUN_DOG="1", STUB_THRESHOLD=json.dumps(threshold))
        if version is not None:
            env["XRAY_DATA"] = version
        proc = subprocess.run(["bash", "run.sh", mode], cwd=root, env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
        return proc, calls

    def test_default_v2_paths_and_validation_threshold(self):
        proc, calls = self.run_pipeline(None)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(all(c["data"] == "xray_v2" for c in calls))
        self.assertEqual(calls[0]["args"], ["scripts/xray_prepare_v2.py"])
        for call in calls:
            args = call["args"]
            if args[0] == "scripts/xray_eval.py" and args[args.index("--split") + 1] == "test":
                self.assertEqual(args[args.index("--thr") + 1], "0.42")
        predict = next(c["args"] for c in calls if c["args"][0] == "scripts/xray_predict.py")
        self.assertEqual(predict[predict.index("--images") + 1], "data/xray_v2/images/val_fakemask")
        self.assertEqual(calls[-1]["args"][-2:], ["--thr", "0.42"])

    def test_explicit_v1_paths(self):
        proc, calls = self.run_pipeline("xray_v1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(calls[0]["args"], ["scripts/xray_prepare.py"])
        self.assertTrue(all(c["data"] == "xray_v1" for c in calls))

    def test_evaluate_does_not_train_or_predict(self):
        proc, calls = self.run_pipeline(None, "evaluate")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual({c["args"][0] for c in calls}, {"scripts/xray_check_dataset.py", "scripts/xray_eval.py"})

    def test_missing_validation_threshold_never_tunes_on_test(self):
        proc, calls = self.run_pipeline(None, "evaluate", threshold=None)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(any("test" in c["args"] for c in calls))

    def test_yaml_cannot_silently_point_at_other_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / "data.yaml").write_text("path: data/xray_v1\n")
            with self.assertRaisesRegex(ValueError, "폴더"):
                check_dataset(data)

    def test_yaml_cannot_swap_validation_and_test(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / "data.yaml").write_text(f"path: {data}\ntrain: images/train\nval: images/test\ntest: images/val\n")
            with self.assertRaisesRegex(ValueError, "분할"):
                check_dataset(data)


if __name__ == "__main__":
    unittest.main()
