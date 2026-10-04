# X-ray 이물 탐지와 검출 한계 분석

식품 포장 라인 X-ray 검사기 사진에서 이물을 찾는 모델을 만들고, 어떤 조건에서 놓치는지 분석해서 현장 판정 기준과 재검사 기준을 제안하는 프로젝트입니다. 2026 제6회 K-인공지능 제조데이터 분석 경진대회 출품용이고, 제출 마감은 10월 8일입니다.

처음에는 [문서 안내](docs/README.md)를 엽니다. 실제 사진이 있는 [데이터 전처리](docs/19_데이터전처리_정리본.md), [모델과 평가](docs/20_모델선정과_평가_정리본.md), [계획과 작업 이력](docs/21_진행계획과_작업이력.md), [대회 규칙](docs/22_대회규칙과_제출준비.md) 네 정리본으로 통합했습니다. 사람과 AI 도구가 같이 지키는 규칙은 [AGENTS.md](AGENTS.md)에 있습니다.

2026-09-30부터 AI 작업은 Codex가 전담하며 `papuagigi` 브랜치에서 이어 갑니다. 현재 상태와 이후 작업 기록은 [계획과 작업 이력](docs/21_진행계획과_작업이력.md) 한 파일에서 관리합니다. 이전 문서 18개 원문은 `archive/문서정리_20261002/`의 모음과 검증된 ZIP에 로컬 보존했습니다.

## 준비

```bash
bash setup.sh
```

uv가 설치되어 있어야 합니다(`uv --version`으로 확인). `setup.sh`는 `.python-version`에 고정한 Python 3.12.14로 `.venv`를 만들고, 직접 의존성과 잠금 목록을 함께 적용합니다. 기존 환경의 Python 버전이 다르면 덮어쓰지 않고 중단합니다.

현재 환경은 Python 3.12.14, PyTorch 2.8.0, torchvision 0.23.0, RF-DETR 1.11.1입니다. 2026-10-04 네 모델의 공통 20epoch 학습과 검증 선택 저장본 테스트를 완료했습니다. 학습 사진 2,453장·검증 107장·테스트 97장을 사용했습니다. 선택 epoch는 YOLOv8n 20, Faster R-CNN 17, RF-DETR-S 8, D-FINE-S 11입니다. 결과표·전체 epoch 선택 기록·실패 조건·같은 MPS 속도는 [공유용 근거](docs/evidence/common_epoch_20261004/README.md)에 있습니다. 최종 모델과 운영 임계값은 미확정입니다. 실제 정상 제품 0장과 기존 테스트 열람 이력을 함께 기록했습니다.

중단 복구 기능은 CPU와 실제 MPS·T4·Drive의 작은 실행에서 확인했습니다. 최근 정상 전체 학습 상태 2개와 epoch별 모델을 보관합니다. 데이터·설정·코드가 달라지면 재개를 거부합니다. 같은 실행 명령에 `--resume`을 사용합니다. Colab GPU 자동 재연결 기능은 아닙니다. 재개할 때는 학습에 사용한 코드·설정·환경을 유지합니다.

새 Colab 동반 노트북은 `notebooks/xray_common_epoch_colab.ipynb`입니다. 저장된 셀의 JSON·구문은 검사했으며, 같은 설치·복구 절차를 현재 T4 세션에서 확인했습니다. 이 노트북 전체를 새 VM에서 처음부터 재현한 검사는 아직 하지 않았습니다. 과거 ZIP·노트북은 새 실험용으로 바로 실행하지 않습니다.

`requirements.txt`는 직접 의존성, `requirements.lock.txt`는 Mac 설치 환경의 잠금 목록입니다. 다른 OS·CUDA 환경은 별도 호환성 검사가 필요합니다. 이전 Colab 예제는 `notebooks/xray_colab_t4.ipynb`, 재개 가능한 학습 코드는 `scripts/xray_train_resumable.py`입니다. 원본 자료·모델·전체 예측 CSV는 Git에 포함하지 않습니다. 아래 명령만으로 이미 완료된 모델 결과가 자동 복원되는 것은 아닙니다.

데이터는 저장소에 없습니다. KAMP 포털 공지(제6회 경진대회 과제공개)에서 `4. X-ray 검사장비 AI 데이터셋.zip`을 내려받아 압축을 풀고, 프로젝트 루트에 `제조AI데이터셋/4. X-ray 검사장비 AI 데이터셋/dataset/` 구조가 되도록 두면 됩니다. 코드는 이 폴더를 읽기만 합니다.

Apple 칩 Mac은 GPU를 자동으로 씁니다. NVIDIA GPU가 있으면 CUDA를, 둘 다 없으면 CPU를 씁니다.

## 실행 (색 네모 제거·고정 분할, 내부 경로 xray_v2)

아래 `run.sh`는 기존 YOLO·마스킹 대조를 재현하는 경로이며 `legacy/custom_ap_v2` 채점입니다. **현재 비교 주지표는 IoU 0.5 F1**입니다. 저장 예측을 현재 기준으로 채점할 때는 `scripts/xray_eval.py --matching iou50`을 사용하고 테스트에는 검증에서 정한 `--thr`를 전달합니다. `scripts/xray_condition_audit.py`는 저장 검증 예측을 재채점하는 조건 분석이며 새 학습을 실행하지 않습니다.

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

[EDA와 네 모델 결과를 반영한 제출 검토본](docs/submission_review_20261004/README.md)을 확인할 수 있습니다. 최종 제출본과 구분합니다.
