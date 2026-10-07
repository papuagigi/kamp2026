"""Validate recorded visual decisions and export separate, traceable YOLO labels.

No detector inference or original colored images are used. An unresolved image
is excluded as a whole: its obvious boxes are drafts, not a complete train label.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import cv2

from xray_direct_label_geometry import GEOMETRY_VERSION, measured_box

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/direct_labeling_20261006"
OUT = ROOT / "data/xray_direct_labels_20261006"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_reviews():
    records = {}
    for path in sorted(SOURCE.glob("reviewer_[abc].jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            index = r["index"]
            if index in records:
                raise ValueError(f"Duplicate review index {index}: {path}:{number}")
            r["review_file"] = str(path.relative_to(ROOT))
            records[index] = r
    for path in sorted(SOURCE.glob("reviewer_*_corrections.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            correction = json.loads(line)
            index = correction["index"]
            if index not in records:
                raise ValueError(f"Correction without original visual review: {index}")
            original = dict(records[index])
            records[index].update(correction)
            records[index]["previous_visual_choice"] = original
            records[index]["correction_file"] = str(path.relative_to(ROOT))
    return records


def export(require_complete=False):
    candidates = json.loads((SOURCE / "candidates.json").read_text())
    reviews = read_reviews()
    assert len(candidates) == 1900
    assert all(r["index"] == i for i, r in enumerate(candidates))
    assert set(reviews) <= set(range(1900))
    if require_complete:
        assert len(reviews) == 1900, "Visual review coverage is incomplete"
    inventory = list(csv.DictReader((ROOT / "data/xray_roadmap_20261002/inventory.csv").open()))
    allowed = {r["stem"] for r in inventory if r["status"] == "pseudo_candidate"}
    excluded_groups = {int(r["group"]) for r in inventory if r["split"] in ("val", "test")}
    assert all(r["stem"] in allowed and r["group"] not in excluded_groups for r in candidates)
    qa_path = SOURCE / "geometry_visual_qa.json"
    qa = json.loads(qa_path.read_text()) if qa_path.exists() else {}
    geometry_checked = qa.get("version") == GEOMETRY_VERSION and qa.get("sample_visual_check") == "passed"
    overrides_path = SOURCE / "box_overrides.json"
    overrides = {(r["index"], r["candidate_id"]): r for r in
                 (json.loads(overrides_path.read_text()) if overrides_path.exists() else [])}
    for folder in ("draft_labels", "reviewed_labels"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    rows, full_records, issue_records = [], [], []
    for index, review in sorted(reviews.items()):
        item = candidates[index]
        assert review["status"] in ("visually_labeled", "needs_review")
        assert len(review["keep"]) == len(set(review["keep"])), (index, "duplicate selection")
        source = ROOT / item["image"]
        assert sha(source) == item["sha256"], (index, "clean source changed")
        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        assert gray.shape == (item["height"], item["width"])
        choices = {q["id"]: q for q in item["candidates"]}
        boxes, geometry_flags = [], []
        for selection in review["keep"]:
            q = choices[selection]
            if (index, selection) in overrides:
                box, flags = overrides[index, selection]["box"], []
            else:
                box, flags = measured_box(gray, q["cx"], q["cy"])
            boxes.append(box)
            geometry_flags.extend(f"candidate_{selection}:{x}" for x in flags)
        for box in review.get("manual_boxes", []):
            boxes.append(list(map(float, box)))
        reasons = []
        if review["status"] == "needs_review" and review["reason"] != "bbox_boundary_only":
            reasons.append(review["reason"])
        if not geometry_checked:
            reasons.append("geometry_sample_check_pending")
        reasons.extend(geometry_flags)
        if not boxes:
            reasons.append("no_confirmed_box_not_normal_label")
        lines = []
        for box in boxes:
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= item["width"] and 0 <= y1 < y2 <= item["height"], (index, box)
            cx, cy = (x1+x2)/2/item["width"], (y1+y2)/2/item["height"]
            width, height = (x2-x1)/item["width"], (y2-y1)/item["height"]
            lines.append(f"0 {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
        draft = OUT / "draft_labels" / (item["stem"]+".txt")
        # No empty file is emitted: an empty detector-style label means negative.
        if lines:
            draft.write_text("\n".join(lines)+"\n")
        elif draft.exists():
            draft.unlink()
        eligible = not reasons
        accepted = OUT / "reviewed_labels" / (item["stem"]+".txt")
        if eligible:
            accepted.write_text("\n".join(lines)+"\n")
        elif accepted.exists():
            accepted.unlink()  # Only this new version's superseded derived file.
        row = {"index": index, "stem": item["stem"], "image": item["image"],
               "group": item["group"], "n_boxes": len(boxes), "source_sha256": item["sha256"],
               "visual_status": review["status"], "visual_reason": review["reason"],
               "export_status": "reviewed_training_candidate" if eligible else "needs_review",
               "review_reason": "|".join(reasons), "review_file": review["review_file"],
               "label": str(accepted.relative_to(ROOT)) if eligible else "",
               "reviewer_model": review.get("reviewer_model", "inherited_not_exposed"),
               "reviewer_effort": review.get("reviewer_effort", "inherited_not_exposed")}
        rows.append(row)
        applied_overrides = [overrides[index, selection] for selection in review["keep"]
                             if (index, selection) in overrides]
        full_records.append(dict(row, boxes=boxes, visual_choice=review,
                                 box_overrides=applied_overrides, geometry_version=GEOMETRY_VERSION))
        if reasons:
            issue_records.append(dict(index=index, stem=item["stem"], reasons=reasons, boxes=boxes))
    for name, content in (("manifest.csv", rows), ("review_queue.csv", [r for r in rows if r["export_status"] == "needs_review"])):
        if rows:
            with (OUT/name).open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(content)
    (OUT/"provenance.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in full_records))
    (SOURCE/"geometry_issues.json").write_text(json.dumps(issue_records, ensure_ascii=False, indent=2)+"\n")
    summary = {"observed_at": datetime.now().astimezone().isoformat(), "expected_photos": 1900,
               "visually_reviewed": len(rows), "remaining": 1900-len(rows),
               "status_counts": dict(Counter(r["export_status"] for r in rows)),
               "reviewed_training_boxes": sum(r["n_boxes"] for r in rows if r["export_status"] != "needs_review"),
               "original_review_reasons": dict(Counter(r["visual_reason"] for r in rows)),
               "heldout_group_overlap": 0, "source_hash_checks": len(rows),
               "geometry_version": GEOMETRY_VERSION, "geometry_sample_checked": geometry_checked,
               "candidate_file_sha256": sha(SOURCE/"candidates.json"),
               "official_split_md5": hashlib.md5((ROOT/"data/xray_v2/split.csv").read_bytes()).hexdigest(),
               "official_or_previous_labels_modified": False,
               "trained_on_this_label_version": False, "human_verified": False,
               "note": "Visual AI annotations; geometric checks are not a measured label accuracy."}
    (OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    (SOURCE/"progress.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    export(parser.parse_args().require_complete)
