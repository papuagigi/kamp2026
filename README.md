**팀 회의 비교(2026-10-06):** [두 프로젝트 비교 PDF](docs/team_compare_20261006/두_프로젝트_비교_회의자료.pdf). 코드와 v7 보고서를 대조한 상세 비교는 [모델 정리본 18.26절](docs/20_모델선정과_평가_정리본.md)에 있습니다.

**최신 상태(2026-10-07): YOLO11s·RF-DETR-S 전체 사진 검사.** 새 AI 라벨과 증강으로 만든 학습 12,161장으로 두 모델의 20epoch 학습을 완료했습니다. 현재 [27쪽 보고서와 편집 원본](docs/submission_review_20261004/README.md), [중심 위치 평가 근거](docs/evidence/center_evaluation_20261007/README.md)를 공유합니다. 구버전 보고서·발표자료는 현재 파일 목록에서 정리했고 로컬 보존본과 Git 이력에서 복원할 수 있습니다.

# X-ray 이물 탐지와 검출 한계 분석

식품 포장 라인 X-ray 검사기 사진에서 이물을 찾는 모델을 만들고, 어떤 조건에서 놓치는지 분석해서 현장 판정 기준과 재검사 기준을 제안하는 프로젝트입니다. 2026 제6회 K-인공지능 제조데이터 분석 경진대회 출품용이고, 제출 마감은 10월 8일입니다.

처음에는 [문서 안내](docs/README.md)를 엽니다. 실제 사진이 있는 [데이터 전처리](docs/19_데이터전처리_정리본.md), [모델과 평가](docs/20_모델선정과_평가_정리본.md), [계획과 작업 이력](docs/21_진행계획과_작업이력.md), [대회 규칙](docs/22_대회규칙과_제출준비.md) 네 정리본으로 통합했습니다. 사람과 AI 도구가 같이 지키는 규칙은 [AGENTS.md](AGENTS.md)에 있습니다.

2026-09-30부터 AI 작업은 Codex가 전담하며 `papuagigi` 브랜치에서 이어 갑니다. 현재 상태와 이후 작업 기록은 [계획과 작업 이력](docs/21_진행계획과_작업이력.md) 한 파일에서 관리합니다. 이전 문서 18개 원문은 `archive/문서정리_20261002/`의 모음과 검증된 ZIP에 로컬 보존했습니다.

## 준비

```bash
bash setup.sh
```

uv가 설치되어 있어야 합니다(`uv --version`으로 확인). `setup.sh`는 `.python-version`에 고정한 Python 3.12.14로 `.venv`를 만들고, 직접 의존성과 잠금 목록을 함께 적용합니다. 기존 환경의 Python 버전이 다르면 덮어쓰지 않고 중단합니다.

현재 기본 환경은 Python 3.12.14, PyTorch 2.8.0, torchvision 0.23.0, RF-DETR 1.11.1입니다. 새 실험은 학습 12,161장·검증 107장·테스트 97장을 사용했습니다. 검증으로 선택한 저장본은 YOLO11s 2epoch, RF-DETR-S 10epoch입니다. [학습·선택 기록](docs/evidence/direct_training_20261006/README.md)과 [위치 평가 결과](docs/evidence/center_evaluation_20261007/README.md)를 구분했습니다.

위치 평가는 기존 테스트를 확인한 뒤 추가한 사후 분석입니다. 선택 저장본·탐지 임계값·공식 평가 TXT를 유지하고, 중심 거리 2·4·6·8픽셀을 모두 계산했습니다. 박스 기반 이전 점수는 내부 연구 기록으로 보존합니다. 실제 정상 제품은 평가에 없으므로 현장 오경보율이나 자동 PASS 안전성을 입증한 결과는 아닙니다.

중단 복구 기능은 CPU와 실제 MPS·T4·Drive의 작은 실행에서 확인했습니다. 최근 정상 전체 학습 상태 2개와 epoch별 모델을 보관합니다. 데이터·설정·코드가 달라지면 재개를 거부합니다. 같은 실행 명령에 `--resume`을 사용합니다. Colab GPU 자동 재연결 기능은 아닙니다. 재개할 때는 학습에 사용한 코드·설정·환경을 유지합니다.

새 라벨 학습에 사용한 Colab 노트북은 `notebooks/xray_direct_training_colab.ipynb`입니다. 기존 GPU 중단 뒤 CPU·MPS 복구 코드도 보존합니다. 현재 정상 학습을 중복 실행하지 말고, 해당 저장본의 코드·환경·데이터 해시가 맞는 복구 절차를 사용합니다. 이전 공통 비교용 `xray_common_epoch_colab.ipynb`와 과거 ZIP은 다른 실험입니다.

`requirements.txt`는 직접 의존성, `requirements.lock.txt`는 Mac 설치 환경의 잠금 목록입니다. 다른 OS·CUDA 환경은 별도 호환성 검사가 필요합니다. 이전 Colab 예제는 `notebooks/xray_colab_t4.ipynb`, 재개 가능한 학습 코드는 `scripts/xray_train_resumable.py`입니다. 원본 자료·모델·전체 예측 CSV는 Git에 포함하지 않습니다. 아래 명령만으로 이미 완료된 모델 결과가 자동 복원되는 것은 아닙니다.

데이터는 저장소에 없습니다. KAMP 포털 공지(제6회 경진대회 과제공개)에서 `4. X-ray 검사장비 AI 데이터셋.zip`을 내려받아 압축을 풀고, 프로젝트 루트에 `제조AI데이터셋/4. X-ray 검사장비 AI 데이터셋/dataset/` 구조가 되도록 두면 됩니다. 코드는 이 폴더를 읽기만 합니다.

Apple 칩 Mac은 GPU를 자동으로 씁니다. NVIDIA GPU가 있으면 CUDA를, 둘 다 없으면 CPU를 씁니다.

## 새 라벨 학습과 현재 평가의 재현 경로

| 단계 | 코드와 필요한 자료 |
|---|---|
| AI 판독 보조·좌표 생성 | `xray_direct_label_review.py`, `xray_direct_label_geometry.py`, `xray_direct_label_export.py`. 실제 시각 판독 결정 기록이 별도로 필요합니다. |
| 학습 증강 자료 생성 | `xray_prepare_direct_training.py`. 라벨 제공 사진과 추가 AI 라벨을 사용하며 검증·테스트는 증강하지 않습니다. |
| 학습·선택 저장본 평가 | `xray_train_direct.py`, `xray_evaluate_direct.py`. 기존 run과 구분한 이름, 데이터·사전학습 가중치가 필요합니다. |
| 중단 복구 | `xray_direct_queue.py`, `xray_resume_direct_cpu.py`, `xray_import_direct_mps.py`, `xray_resume_direct_mps.py`. 장치 전환 기록과 정상 전체 저장본을 확인합니다. |
| 신뢰성·실패·속도 분석 | `xray_audit_direct_training.py`, `xray_rescore_direct.py`, `xray_analyze_direct_errors.py`, `xray_benchmark_direct_pair.py` |
| 중심 위치 재채점 | `xray_center_analysis.py`. 고정 예측 6개·공식 TXT·분할·protocol·이전 재채점 요약을 읽습니다. 새 학습을 실행하지 않습니다. |

모든 스크립트는 `scripts/` 아래에 있습니다. 현재 저장소는 코드와 작은 근거를 공유하며, 전체 영상·AI TXT·가중치·예측 CSV를 포함하지 않습니다. 코드만 내려받았다고 학습 결과가 자동 복원되지는 않습니다. 필요한 입력과 해시는 각 근거 묶음 README와 source manifest에 있습니다.

중심 위치 분석을 재현하려면 필요한 원본 예측·평가 자료를 준비한 뒤 다음과 같이 규칙과 기존 요약을 복원합니다. 기존 분석 폴더가 있으면 먼저 파일 해시가 같은지 확인합니다.

```bash
mkdir -p reports/center_evaluation_20261007 reports/direct_analysis_20261007
cp docs/evidence/center_evaluation_20261007/protocol.json reports/center_evaluation_20261007/protocol.json
cp docs/evidence/direct_training_20261006/rescore_summary.csv reports/direct_analysis_20261007/rescore_summary.csv
.venv/bin/python scripts/test_xray_eval.py
.venv/bin/python scripts/xray_center_analysis.py
```

## 실행 (색 네모 제거·고정 분할, 내부 경로 xray_v2)

아래 `run.sh`는 **과거 YOLOv8n·마스킹 대조 실험**을 재현하는 경로이며 `legacy/custom_ap_v2` 채점입니다. 현재 제출 보고서의 새 라벨 학습은 다음 절의 스크립트를 사용했습니다. 현재 위치 평가는 `scripts/xray_eval.py --matching center2` 또는 `center4`, `center6`, `center8`이며 테스트에는 검증에서 고정한 `--thr`를 전달합니다. 과거 `iou50`·`iou75`·`legacy` 모드는 연구 이력 재현용으로 유지했습니다.

2026-10-01부터 준비·학습·예측·채점·마스킹 대조의 기본 기준은 v2입니다. 전처리는 [사진 설명](docs/19_데이터전처리_정리본.md), 채점은 [모델과 평가 설명](docs/20_모델선정과_평가_정리본.md)을 읽습니다.

```bash
# 설정 확인만 (학습하지 않음)
bash run.sh dry-run
# 데이터 준비·고정 분할 확인만
bash run.sh prepare
# 저장된 YOLO 예측 재채점만 (학습·추론하지 않음)
bash run.sh evaluate
# 기존 DoG도 같은 검증 임계값 원칙으로 재채점
RUN_DOG=1 bash run.sh evaluate
# 새로운 이름으로 YOLO 학습부터 마스킹 대조까지 실행
NAME=v2_yolov8n_640_new bash run.sh
```

순서대로 다음을 실행합니다.

1. `scripts/xray_prepare_v2.py`: 색 네모를 지우고 촬영 묶음 단위로 분할합니다. 기존 데이터가 있으면 재생성을 생략합니다.
2. `scripts/xray_check_dataset.py`: 파일 목록·분할 확인값·촬영 묶음 누수를 확인합니다.
3. `scripts/xray_train_yolo.py`: YOLO를 학습하고 검증·평가 예측을 저장합니다. 장치는 MPS → CUDA → CPU 순으로 고릅니다. 기존 이름의 학습 결과는 덮어쓰지 않습니다. DoG는 `RUN_DOG=1`일 때만 추가 실행합니다.
4. `scripts/xray_eval.py`: 검증에서 임계값을 정하고 평가에는 같은 값을 적용합니다. 새 JSON에는 `custom_ap_v2`가 붙으며 과거 채점 파일을 보존합니다.
5. 가짜 마스킹 사진 생성 → `scripts/xray_predict.py` 예측 → `scripts/xray_fakemask_eval.py` 채점도 같은 데이터와 검증 임계값을 씁니다.

과거 v1은 `XRAY_DATA=xray_v1 bash run.sh evaluate`처럼 명시적으로 선택합니다. 새 데이터 준비가 필요한 경우에만 `prepare` 또는 전체 실행을 사용합니다.

### v2 데이터 (2026-09-30부터 기준)

```bash
.venv/bin/python scripts/xray_prepare_v2.py
```

`data/xray_v2/`를 만듭니다. 색 네모(팔레트 244~255번)만 지우고 나비에-스토크스 인페인팅으로 메우며, 점검 묶음 단위로 호기 × 시험편 종류마다 60:20:20을 목표로 나눕니다. 모든 현행 스크립트는 기본으로 v2를 읽습니다. 현재 처리와 분할 설명은 [데이터 정리본](docs/19_데이터전처리_정리본.md)에 있습니다.

전처리와 기존 탐지 지표를 다시 확인하려면 다음 명령을 씁니다. 전처리는 임시 폴더에서 다시 만들고 채점은 저장 예측으로 수행하며, 결과는 `reports/handover_20261001/`에 둡니다. 수정된 AP와 과거 AP는 나란히 보존하고 AP 외 지표의 일치를 검사합니다. 학습과 추론은 다시 실행하지 않습니다.

```bash
.venv/bin/python scripts/xray_handover_check.py
.venv/bin/python -m unittest discover -s scripts -p 'test_xray_*.py' -v
```

그 밖에 `scripts/xray_object_removed.py`는 이물을 지운 자리에서도 탐지가 나오는지 보는 검사이고, `scripts/xray_predict.py`는 학습된 모델로 임의 폴더를 예측하며, `scripts/xray_condition_plot.py`는 여러 모델의 비교표와 그림을 만듭니다.

## 폴더

- `scripts/` 실행 스크립트
- `docs/` 주제별 정리본 네 개와 시작 안내. 대회 안내문과 보고서 양식은 `docs/대회자료/`에 있습니다(원문은 저장소에 올리지 않습니다). 과거 문서 원문은 `archive/문서정리_20261002/`에 로컬 보존합니다.
- `data/`, `reports/`, `runs/` 스크립트가 만드는 가공 데이터, 채점 결과, 학습 결과. 저장소에 올리지 않습니다.
- `outputs/` 제출용 예측결과 파일
- `weights/` 사전학습 가중치. 처음 학습할 때 Ultralytics가 자동으로 내려받습니다.
- `제조AI데이터셋/` 대회 원본 데이터. 저장소에 올리지 않습니다.

## 파일 형식

- 예측 CSV: `stem, cx, cy, w, h, score`. 픽셀 단위이고 한 줄이 박스 하나입니다.
- 데이터 목록 `data/xray_v2/manifest.csv`: 사진 경로·원본 MD5·호기·촬영 시각·시험편 종류·촬영 묶음·분할·마스킹 픽셀 수 등. 고정 분할은 `split.csv`입니다.
- 채점 JSON: 기존 지표와 `metric_version, data_version, split_csv_md5, threshold_source, tp, fp, fn`. `ap`는 자체 매칭의 보간 PR 면적이며 COCO AP와 다릅니다. 정답 없는 세트의 AP·재현율·F1과 검출 없는 경우의 정밀도는 `null`입니다. 정상 사진의 오탐은 `image_false_alarm_rate`로 따로 봅니다.

[현재 제출 검토본](docs/submission_review_20261004/README.md)을 확인할 수 있습니다. 최종 제출본과 구분합니다.
