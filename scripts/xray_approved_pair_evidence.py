"""Export evidence for the approved model pair; do not train or tune models."""
import csv
import json
from pathlib import Path

from xray_recovery import sha256


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'reports/design_review_20261004/yolo8_rfdetr_all_20261005'
    dest = root / 'docs/evidence/approved_pair_20261005'
    dest.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / 'summary.json').read_text())
    execution = json.loads((source / 'execution.json').read_text())
    assert summary['execution_sha256'] == sha256(source / 'execution.json')
    assert execution['status'] == 'complete'
    assert execution['policy_sha256'] == sha256(source / 'policy.json')
    for name, digest in execution['csv_sha256'].items():
        assert sha256(source / name) == digest
    first_pass = [c for c in execution['calls'] if c['repeat'] == 0]
    assert len(first_pass) == 107
    no_alarm = sum(not c['first_predictions'] and not c['second_predictions'] for c in first_pass)
    rows = list(csv.DictReader((root / 'docs/evidence/common_epoch_20261004/comparison.csv').open()))
    evidence = {
        'date': '2026-10-05', 'models_approved': ['YOLOv8n', 'RF-DETR-S'],
        'approved_conflict_action': 'preserve both predictions and hold for reinspection',
        'input_route_status': 'approved: both models inspect every full image; crops/tiles remain exploratory',
        'measurement_route': 'both models independently infer every full validation image',
        'validation_images': 107, 'validation_gt_boxes': 243, 'repeats': 3,
        'unique_validation_images': 107, 'pair_no_alarm_images_first_repeat': no_alarm,
        'pair_with_any_alarm_first_repeat': 107-no_alarm,
        'photo_alarm_note': 'alarm presence does not establish correct localization or product safety',
        'metrics': summary,
        'historical_individual_test': {'images': 97, 'gt_boxes': 225, 'rows': rows},
        'normal_product_images': 0, 'production_accuracy': None, 'operational_f1': None,
        'referred_field_meaning': 'second model called; not a measured factory hold rate',
        'training_this_export': False, 'new_test_this_export': False,
        'source_hashes': {str(p.relative_to(root)): sha256(p) for p in [
            source/'summary.json', source/'execution.json', source/'policy.json',
            root/'docs/evidence/common_epoch_20261004/comparison.csv',
            root/'reports/review_roles_20261005/summary.json']},
    }
    (dest/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    (dest/'live_summary.json').write_bytes((source/'summary.json').read_bytes())
    with (dest/'image_decisions.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['repeat','stem','first_boxes','second_boxes','state','error','milliseconds'], lineterminator='\n')
        w.writeheader()
        for c in execution['calls']:
            w.writerow(dict(repeat=c['repeat'],stem=c['stem'],first_boxes=len(c['first_predictions']),
                            second_boxes=len(c['second_predictions'] or []),state=c['state'],
                            error=c['error'],milliseconds=c['seconds']*1000))
    print(json.dumps({'unique_images':107,'any_alarm_images':107-no_alarm,
                      'mean_ms':summary['mean_ms'],'p95_ms':summary['p95_ms']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
