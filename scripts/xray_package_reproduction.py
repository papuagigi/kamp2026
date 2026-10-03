"""Package relative-path code/data/results for retraining; source assets stay unchanged."""
import ast
import hashlib
import json
import unicodedata
from pathlib import Path
import zipfile
import pandas as pd
import yaml
from xray_config import ROOT,DATA

REPORT=ROOT/'reports/roadmap_20261002'
PREP=ROOT/'data/xray_roadmap_20261002'
OUT=ROOT/'outputs/submission_draft'


def nfc(s):return unicodedata.normalize('NFC',s)


def main():
    final=json.loads((REPORT/'final_comparison.json').read_text())
    selection=json.loads((REPORT/'model_selection.json').read_text())
    target=OUT/'코드_데이터_재학습패키지.zip'
    OUT.mkdir(exist_ok=True,parents=True)
    files=set();overrides={};catalog=[]
    seeds=['xray_check_dataset','xray_run_roadmap_queue','xray_benchmark_models','test_xray_eval','test_xray_pipeline','xray_train_controlled','xray_predict_controlled','xray_eval','xray_validate_roadmap',
           'xray_prepare_roadmap','xray_refine_removal','xray_prepare_teacher_split','xray_pseudo_label',
           'xray_restore_variants','xray_download_pretrained','xray_product_outputs']
    # Include exact local import dependencies without unrelated drafts/logs.
    todo=list(seeds)
    while todo:
        name=todo.pop();p=ROOT/'scripts'/f'{name}.py'
        if p in files:continue
        files.add(p)
        for node in ast.walk(ast.parse(p.read_text())):
            mods=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            for m in mods:
                if m and (ROOT/'scripts'/f'{m}.py').exists():todo.append(m)
    for name in ['requirements.txt','requirements.lock.txt','.python-version','setup.sh']:files.add(ROOT/name)
    for split in ['train','val','test']:
        for kind in ['images','labels']:files.update((DATA/kind/split).glob('*'))
    for name in ['manifest.csv','split.csv','summary.json']:files.add(DATA/name)
    overrides['data/xray_v2/data.yaml']=yaml.safe_dump({'path':'data/xray_v2','train':'images/train','val':'images/val','test':'images/test','names':{0:'defect'}}).encode()
    for folder in ['images','labels','pseudo_labels']:
        if (PREP/folder).exists():files.update(p for p in (PREP/folder).rglob('*') if p.is_file())
    files.update(p for p in PREP.iterdir() if p.is_file() and p.suffix in {'.csv','.json'})
    for v in (PREP/'variants').iterdir():
        if not v.is_dir():continue
        files.add(v/'data.yaml')
        for kind in ['images','labels']:
            for p in (v/kind).rglob('*'):
                if p.is_file():catalog.append({'destination':str(p.relative_to(ROOT)),
                                               'source':str(p.resolve().relative_to(ROOT))})
    overrides['data/xray_roadmap_20261002/variants_catalog.json']=json.dumps(catalog,ensure_ascii=False,indent=2).encode()
    inv=pd.read_csv(PREP/'inventory.csv')
    files.update(ROOT/p for p in inv.source)
    # Original official labels are included for traceability; the original folder itself is read-only.
    from xray_prepare_v2 import LAB
    files.update(LAB.glob('*.txt'))
    for name in ['unique_raw_groups.csv','unlabeled_inventory.csv']:files.add(ROOT/'reports/augmentation_options_study_20261002'/name)
    for name in ['final_comparison.json','official_baseline_comparison.json','ablation_results.json','dataset_selection.json',
                 'model_selection.json','inference_benchmark.json','teacher_split.json','teacher_calibration_stems.txt','preprocessing_validation.json','evaluation_policy_20261002.json','official_test_comparison.json']:
        files.add(REPORT/name)
    for name in ['summary.json','calibration.json','models.json','decisions.csv','accepted.csv']:
        files.add(REPORT/'pseudo'/name)
    for row in final:
        for split in ['val','test']:
            stem=f"preds_{row['run']}_{split}"
            for suffix in ['.csv','.run.json','_iou50_v1_eval.json']:files.add(ROOT/'reports'/f'{stem}{suffix}')
        meta=json.loads((ROOT/'runs'/row['run']/'execution.json').read_text())
        overrides[f"saved_execution/{row['run']}.json"]=json.dumps(meta,indent=2).encode()
    files.update(p for p in (ROOT/'outputs/roadmap_20261002').glob('*') if p.is_file())
    variant=selection['dataset_variant'];recommended=next(r for r in final if r['run']==selection['recommended_run'])
    commands=[]
    for row in final:
        meta=json.loads((ROOT/'runs'/row['run']/'execution.json').read_text());kind=meta['model'];name='reproduce_'+kind
        cmd=f'.venv/bin/python scripts/xray_train_controlled.py --model {kind} --variant {variant} --name {name} --epochs 20 --batch {meta["batch"]} --device auto'
        if meta.get('steps'):cmd+=f' --steps {meta["steps"]}'
        commands.extend([cmd,f'.venv/bin/python scripts/xray_predict_controlled.py --run {name} --split val',
            f'.venv/bin/python scripts/xray_eval.py reports/preds_{name}_val.csv --split val --matching iou50'])
    readme=f'''# X-ray 이물 탐지 재학습 패키지

## 포함 범위

실제 사용한 처리 사진·공식 TXT·생성 TXT·고정 분할·추가 사진 원본·실행 코드·설정·공통 채점 결과·테스트 예측을 담았습니다. 실험에서 생성한 모든 데이터 조건을 복원할 수 있습니다. 학습 가중치는 ZIP에 포함하지 않으며 공식 사전학습 가중치를 내려받아 재학습합니다. 선정 후보는 {selection['model']}, 데이터 조건은 `{variant}`입니다.

## 준비

1. 이 ZIP을 새 빈 폴더에 풉니다. 원본 작업 폴더 위에 덮어쓰지 마세요.
2. uv가 있는 터미널에서 `bash setup.sh`를 실행합니다. Python 3.12.14·잠금 파일은 macOS 실행 기록입니다. CUDA·Windows 환경은 별도 설치 검증이 필요합니다.
3. `.venv/bin/python scripts/xray_restore_variants.py`로 상대경로 링크를 복원합니다. 링크를 지원하지 않으면 파일을 복사합니다.
4. `.venv/bin/python scripts/xray_check_dataset.py`와 `.venv/bin/python scripts/xray_validate_roadmap.py`로 검사합니다.
5. `.venv/bin/python scripts/xray_download_pretrained.py`로 공식 사전학습 가중치를 받습니다. SHA-256 불일치 시 중단합니다.

## 저장 예측 재채점

학습을 다시 하지 않고도 다음 명령으로 저장 결과를 확인할 수 있습니다. 이는 재학습 재현과 다릅니다.

```sh
.venv/bin/python scripts/xray_eval.py reports/preds_{recommended['run']}_test.csv --split test --matching iou50 --thr {recommended['validation']['thr']} --tag reproduced
```

## 같은 조건으로 재학습

```sh
{chr(10).join(commands)}
```

테스트는 재학습한 모델의 검증 결과 JSON에서 임계값 `thr`를 확인한 뒤 그 값으로 평가하세요. 이전 모델의 임계값을 새 모델에 그대로 적용하지 않습니다. 비교 당시 Faster R-CNN은 CPU, 다른 모델은 MPS에서 학습했으므로 `--device auto`로 다른 장치를 선택하면 학습 결과가 달라질 수 있습니다. 같은 시드라도 장치별 수치가 완전히 같다고 보장하지 않습니다.

## 전처리·추가 라벨 재현

전처리 결과와 각 사진의 부모·좌표 변환은 `data/xray_roadmap_20261002/augmentation.csv`에 있습니다. 현재 ZIP에는 이미 생성된 폴더가 있으므로 준비 스크립트가 덮어쓰기를 거부하는 것이 정상입니다. 전처리 자체를 재생성하려면 별도의 새 추출 폴더에서 처리 폴더를 백업한 후 `xray_prepare_roadmap.py`, `xray_refine_removal.py`, `xray_prepare_teacher_split.py`를 차례대로 실행합니다. 원본 폴더와 `data/xray_v2`는 수정하지 않습니다.

AI 라벨용 모델은 `teacher` 조건에서 YOLO와 RF-DETR을 각각 배치 8·4, 20epoch로 학습한 후 `xray_pseudo_label.py --yolo-run 실행이름 --rfdetr-run 실행이름`으로 생성합니다. 기존 pseudo 결과는 별도 보관해야 합니다. 공식 정답과 추가 AI 라벨은 구분되어 있습니다. 자동 검사를 통과하지 못한 사진은 학습에서 제외합니다.

## 판단 범위

주지표는 `iou50_v1`의 IoU 0.5 이상 일대일 대응 F1입니다. 점수가 높은 예측부터 미대응 정답 중 가장 큰 IoU와 연결합니다. AP는 all-points 방식이며 COCO AP가 아닙니다. 기존 중심 거리 8px 또는 IoU 0.3 결과는 초기 비교로 보존합니다. 테스트 사진은 양성 시험편 중심이고 과거 확인 이력이 있습니다. 실제 정상 제품 오경보율·불량 확률 보정·공장 전체 처리 지연·새 장치 전체 재학습은 검증되지 않았습니다. `outputs/roadmap_20261002`의 판정은 검토용이며 자동 통과 기능은 없습니다.
'''
    overrides['README.md']=readme.encode()
    hashes={}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        used=set()
        for p in sorted(files):
            if not p.is_file():raise FileNotFoundError(p)
            name=nfc(p.relative_to(ROOT).as_posix());content=p.read_bytes()
            if name in overrides:continue
            if p.suffix in {'.py','.md','.txt','.csv','.json','.yaml','.sh'}:
                text=nfc(content.decode('utf-8'))
                text=text.replace(nfc(str(ROOT))+'/', '')
                if '/Users/' in text or 'juyeonkim' in text or '경희대학교' in text:raise ValueError(f'Personal path in export: {name}')
                content=text.encode('utf-8')
            if name in used:raise ValueError('Duplicate archive path: '+name)
            archive.writestr(name,content);used.add(name);hashes[name]=hashlib.sha256(content).hexdigest()
        for name,content in overrides.items():
            name=nfc(name);clean=nfc(content.decode()).replace(nfc(str(ROOT))+'/', '')
            if '/Users/' in clean or 'juyeonkim' in clean or '경희대학교' in clean:raise ValueError(f'Personal path in export override: {name}')
            content=clean.encode()
            if name in used:raise ValueError('Duplicate override: '+name)
            archive.writestr(name,content);used.add(name);hashes[name]=hashlib.sha256(content).hexdigest()
        archive.writestr('PACKAGE_SHA256.json',json.dumps(hashes,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as archive:assert archive.testzip() is None
    result={'zip':target.name,'files':len(hashes),'bytes':target.stat().st_size,'zip_crc':'passed',
            'includes_pretrained_or_trained_weights':False,'retraining_not_rerun_in_fresh_environment':True}
    (OUT/'패키지_검증정보.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result)

if __name__=='__main__':main()
