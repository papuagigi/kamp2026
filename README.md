# X-ray 이물 탐지와 검출 한계 분석

식품 포장 라인 X-ray 검사기 사진에서 이물을 찾는 모델을 만들고, 어떤 조건에서 놓치는지 분석해서 현장 판정 기준과 재검사 기준을 제안하는 프로젝트입니다. 2026 제6회 K-인공지능 제조데이터 분석 경진대회 출품용이고, 제출 마감은 10월 8일입니다.

무엇을 왜 하는지는 [docs/모델선정과_작업계획.md](docs/모델선정과_작업계획.md)에, 날짜별 진행 상황은 [docs/작업일지.md](docs/작업일지.md)에 있습니다. 사람과 AI 도구가 같이 지키는 규칙은 [AGENTS.md](AGENTS.md)에 있습니다.

## 준비

```bash
bash setup.sh
```

`.venv` 가상환경을 만들고 `requirements.txt`의 패키지를 설치한 뒤 학습 장치를 확인합니다. Python 3.9에서 3.13 사이면 됩니다. `requirements.txt`에는 직접 쓰는 패키지의 버전이, `requirements.lock.txt`에는 설치된 모든 패키지의 정확한 버전이 있습니다. 결과가 미묘하게 다를 때는 lock 파일로 설치합니다.

데이터는 저장소에 없습니다. KAMP 포털 공지(제6회 경진대회 과제공개)에서 `4. X-ray 검사장비 AI 데이터셋.zip`을 내려받아 압축을 풀고, 프로젝트 루트에 `제조AI데이터셋/4. X-ray 검사장비 AI 데이터셋/dataset/` 구조가 되도록 두면 됩니다. 코드는 이 폴더를 읽기만 합니다.

Apple 칩 Mac은 GPU를 자동으로 씁니다. NVIDIA GPU가 있으면 CUDA를, 둘 다 없으면 CPU를 씁니다.

## 실행

```bash
bash run.sh
```

순서대로 다음을 실행합니다.

1. `scripts/xray_prepare.py`: 사진의 색 네모를 지우고, 촬영 묶음 단위로 학습·검증·평가를 나누고, 시편 여부를 표시합니다. 결과는 `data/xray_v1/`
2. `scripts/xray_baseline_classical.py`: 딥러닝을 쓰지 않는 기준선. 주변보다 어두운 작은 점을 찾습니다.
3. `scripts/xray_train_yolo.py`: YOLO를 학습하고 검증·평가 사진을 예측합니다.
4. `scripts/xray_eval.py`: 예측을 정답과 맞춰 채점하고 조건별 재현율을 냅니다.
5. `scripts/xray_fake_mask.py`와 `scripts/xray_fakemask_eval.py`: 모델이 지운 자리의 흔적을 배우지 않았는지 확인합니다.

그 밖에 `scripts/xray_object_removed.py`는 이물을 지운 자리에서도 탐지가 나오는지 보는 검사이고, `scripts/xray_predict.py`는 학습된 모델로 임의 폴더를 예측하며, `scripts/xray_condition_plot.py`는 여러 모델의 비교표와 그림을 만듭니다.

## 폴더

- `scripts/` 실행 스크립트
- `docs/` 계획과 작업일지
- `data/`, `reports/`, `runs/` 스크립트가 만드는 가공 데이터, 채점 결과, 학습 결과. 저장소에 올리지 않습니다.
- `제조AI데이터셋/` 대회 원본 데이터. 저장소에 올리지 않습니다.

## 파일 형식

- 예측 CSV: `stem, cx, cy, w, h, score`. 픽셀 단위이고 한 줄이 박스 하나입니다.
- 데이터 목록 `data/xray_v1/manifest.csv`: `stem, path, machine, dt, hour, n_boxes, width, height, burst_id, is_testpiece, split, masked_px`
- 채점 JSON: `precision, recall, f1, ap, fp_per_image, thr_recall99, recall_by_condition`
