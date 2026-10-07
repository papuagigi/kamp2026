"""정답을 손으로 계산할 수 있는 탐지 사례 및 빈 입력에 대한 회귀 검사."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import xray_eval as evaluation


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def score(self, gt_stems, predictions, image_stems=None, thr=None, matching="legacy"):
        gt = pd.DataFrame([dict(stem=s, cx=20., cy=20., w=10., h=10., size=10., edge=50., contrast=40., card_type="막대1", machine="가상", width=100, n_boxes=1) for s in gt_stems], columns=evaluation.GT_COLUMNS)
        manifest = pd.DataFrame({"stem": image_stems if image_stems is not None else sorted(set(gt_stems) | {p[0] for p in predictions})})
        csv = self.root / "pred.csv"
        pd.DataFrame(predictions, columns=evaluation.PRED_COLUMNS).to_csv(csv, index=False)
        with patch.object(evaluation, "load_gt", return_value=(gt, manifest)):
            result, curve, targets = evaluation.evaluate(csv, "val", thr=thr, matching=matching)
        json.dumps(result, allow_nan=False)
        return result

    def box(self, stem, score=0.9):
        return (stem, 20., 20., 10., 10., score)

    def test_perfect_single_many_and_tied(self):
        for count, tied in [(1, False), (5, False), (243, False), (5, True)]:
            with self.subTest(count=count, tied=tied):
                names = [str(i) for i in range(count)]
                result = self.score(names, [self.box(s, 0.9 if tied else 1 - i / count) for i, s in enumerate(names)])
                self.assertAlmostEqual(result["ap"], 1.0)
                self.assertEqual(result["f1"], 1.0)

    def test_partial_recall_area(self):
        result = self.score(["a", "b"], [self.box("a")])
        self.assertAlmostEqual(result["ap"], 0.5)
        self.assertEqual(result["fn"], 1)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_false_positive_before_true_positive(self):
        result = self.score(["a"], [self.box("normal", 0.9), self.box("a", 0.8)])
        self.assertAlmostEqual(result["ap"], 0.5)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["image_false_alarm_rate"], 1)

    def test_tied_true_and_false_positive(self):
        result = self.score(["a"], [self.box("normal"), self.box("a")])
        self.assertAlmostEqual(result["ap"], 0.5)

    def test_duplicate_prediction_is_not_second_true_positive(self):
        result = self.score(["a"], [self.box("a", 0.9), self.box("a", 0.8)], thr=0)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertAlmostEqual(result["ap"], 1.0)

    def test_no_predictions_with_ground_truth(self):
        result = self.score(["a", "b"], [])
        self.assertEqual((result["ap"], result["recall"], result["f1"], result["fn"]), (0., 0., 0., 2))
        self.assertIsNone(result["thr"])
        self.assertIsNone(result["precision"])

    def test_provided_threshold_can_remove_all_predictions(self):
        result = self.score(["a"], [self.box("a")], thr=0.95)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["f1"], 0.)
        self.assertEqual(result["ap"], 1.)  # AP는 운영 임계값과 별개
        self.assertEqual(result["thr"], 0.95)
        self.assertEqual(result["threshold_source"], "provided")

    def test_negative_only_empty_and_false_positive(self):
        for preds, expected_fp in [([], 0), ([self.box("normal")], 1)]:
            with self.subTest(preds=preds):
                result = self.score([], preds, image_stems=["normal"], thr=0.5)
                for key in ["ap", "recall", "f1", "images_fully_detected"]:
                    self.assertIsNone(result[key])
                self.assertEqual(result["fp"], expected_fp)
                self.assertEqual(result["image_false_alarm_rate"], expected_fp)

    def test_all_predictions_wrong(self):
        result = self.score(["a"], [self.box("normal")])
        self.assertEqual((result["ap"], result["tp"], result["fp"], result["fn"]), (0., 0, 1, 1))

    def test_empty_evaluation_set_rejected(self):
        with self.assertRaisesRegex(ValueError, "사진"):
            self.score([], [], image_stems=[])

    def test_invalid_predictions_rejected(self):
        for pred in [self.box("a", float("nan")), ("a", 20., 20., -1., 10., 0.9)]:
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                self.score(["a"], [pred])

    def test_legacy_zero_area_predictions_are_retained_and_counted(self):
        result = self.score(["a"], [self.box("a"), ("normal", 0., 0., 10., 0., 0.1)], thr=0)
        self.assertEqual(result["n_zero_area_predictions"], 1)
        self.assertEqual(result["fp"], 1)

    def test_iou_only_rejects_close_centers_without_overlap(self):
        predictions = [("a", 26., 20., 2., 2., .9)]
        legacy = self.score(["a"], predictions)
        strict = self.score(["a"], predictions, matching="iou50")
        self.assertEqual(legacy["tp"], 1)
        self.assertEqual((strict["tp"], strict["fp"], strict["fn"]), (0, 1, 1))

    def test_iou_half_boundary_and_three_quarters(self):
        predictions = [("a", 20., 20., 20., 10., .9)]  # 100 / 200 = 0.5 exactly
        self.assertEqual(self.score(["a"], predictions, matching="iou50")["tp"], 1)
        self.assertEqual(self.score(["a"], predictions, matching="iou75")["tp"], 0)

    def test_strict_duplicate_and_zero_area(self):
        result = self.score(["a"], [self.box("a"), self.box("a", .8), ("a", 20., 20., 0., 10., .7)], thr=0, matching="iou50")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 2, 0))

    def test_strict_prefers_overlap_instead_of_nearest_center(self):
        gt = pd.DataFrame([dict(stem="a", cx=20., cy=20., w=30., h=10.),
                           dict(stem="a", cx=22., cy=20., w=10., h=10.)])
        preds = pd.DataFrame([self.box("a")], columns=evaluation.PRED_COLUMNS)
        hits, _ = evaluation.match(preds, gt, iou_thr=.3, iou_only=True)
        self.assertEqual(hits.hit.iloc[0], 1)

    def test_strict_test_requires_frozen_threshold(self):
        with self.assertRaisesRegex(ValueError, "임계값"):
            evaluation.evaluate(self.root / "unused.csv", "test", matching="iou50")

    def test_center_distance_boundary_and_box_size_independence(self):
        for size in [2., 100.]:
            result = self.score(["a"], [("a", 22., 20., size, size, .9)], thr=.5, matching="center2")
            self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
            self.assertEqual(result["center_error"]["median_px"], 2.)
        result = self.score(["a"], [("a", 22.0001, 20., 10., 10., .9)], thr=.5, matching="center2")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

    def test_center_no_iou_fallback(self):
        result = self.score(["a"], [("a", 23., 20., 10., 10., .9)], thr=.5, matching="center2")
        self.assertEqual(result["tp"], 0)

    def test_center_duplicates_and_missed_targets_remain_in_counts(self):
        result = self.score(["a", "b"], [self.box("a"), self.box("a", .8)], thr=.5, matching="center4")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["center_error"]["n_matched"], 1)
        result = self.score(["a", "b"], [], thr=.5, matching="center4")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 2))
        self.assertIsNone(result["center_error"]["median_px"])

    def test_center_no_cross_image_match_and_low_score_is_excluded(self):
        result = self.score(["a"], [self.box("b"), self.box("a", .49)], thr=.5, matching="center8")
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        with self.assertRaisesRegex(ValueError, "임계값"):
            evaluation.evaluate(self.root / "unused.csv", "test", matching="center2")

    def test_missing_csv_header_rejected(self):
        csv = self.root / "empty.csv"
        csv.touch()
        with patch.object(evaluation, "load_gt", return_value=(pd.DataFrame(), pd.DataFrame({"stem": ["a"]}))):
            with self.assertRaisesRegex(ValueError, "헤더"):
                evaluation.evaluate(csv, "val")

    def test_load_gt_empty_labels_and_missing_image(self):
        (self.root / "images/val").mkdir(parents=True)
        (self.root / "labels/val").mkdir(parents=True)
        pd.DataFrame([dict(stem="normal", split="val", width=32, height=32, machine="test", n_boxes=0)]).to_csv(self.root / "manifest.csv", index=False)
        (self.root / "labels/val/normal.txt").write_text("")
        cv2.imwrite(str(self.root / "images/val/normal.png"), np.full((32, 32), 210, dtype=np.uint8))
        with patch.object(evaluation, "DATA", self.root):
            gt, man = evaluation.load_gt("val")
            self.assertTrue(gt.empty)
            self.assertIn("stem", gt.columns)
            self.assertEqual(len(man), 1)
            (self.root / "images/val/normal.png").unlink()
            with self.assertRaises(FileNotFoundError):
                evaluation.load_gt("val")


if __name__ == "__main__":
    unittest.main()
