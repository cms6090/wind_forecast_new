---
name: model-tuning
description: 하이퍼파라미터 튜닝 스킬. 05_tuning.ipynb 작성, Optuna·Grid Search·Random Search 사용, LightGBM/XGBoost/CatBoost 파라미터 탐색 범위 설계, 튜닝 결과 해석과 과적합 방지를 다룰 때 반드시 사용한다.
---

# Model Tuning — 하이퍼파라미터 튜닝

## 1. 원칙

- **선택된 유망 모델(1~2개)만** 튜닝한다. 전 모델 튜닝은 시간 낭비
- 튜닝 목적 함수 = **동일 CV 구조의 fold 평균 Score** (src/metric.py). 단일 fold 최적화 금지
- 도구는 Optuna(TPE) 기본. 탐색 횟수보다 탐색 범위 설계가 중요
- 튜닝 전 점수를 고정 기준으로 두고, 최종 개선 폭을 보고한다. 개선이 fold 표준편차 이내면 "튜닝 효과 미미"로 정직하게 기록 (흔한 결과이며 그 자체로 정보)

## 2. LightGBM 탐색 범위 가이드 (출발점)

| 파라미터 | 범위 | 비고 |
|---|---|---|
| learning_rate | 0.01~0.1 (log) | 낮추면 n_estimators↑ + early stopping |
| num_leaves | 15~255 (log) | max_depth와 함께 복잡도 축 |
| min_data_in_leaf | 20~200 | 시계열 노이즈 대응 핵심. 작으면 과적합 |
| feature_fraction | 0.6~1.0 | |
| bagging_fraction / freq | 0.6~1.0 / 1 | |
| lambda_l1, lambda_l2 | 0~10 (log) | |

- `n_estimators`는 튜닝하지 않고 early stopping(예: 200 rounds)으로 결정
- early stopping의 valid는 **각 fold의 검증 구간** 사용 (별도 무작위 분할 금지 — 누수)

## 3. 절차

1. 1차 coarse 탐색: 넓은 범위 × 50~100 trial
2. 상위 trial 파라미터 분포 확인 → 범위 좁혀 2차 탐색
3. 최적 파라미터를 seed 3개로 재검증 (안정성 확인)
4. `study` 결과와 최종 파라미터를 노트북에 명시 + dict로 저장 (train.ipynb에서 하드코딩으로 재사용 — 재현성)

## 4. 과적합 방지 장치

- 튜닝 중 train-valid 격차를 모니터링. 격차가 큰 파라미터 조합은 상위여도 배제 고려
- 튜닝 후 리더보드 1회 제출로 로컬-리더보드 상관 확인 (일일 5회 제한 관리)
- "튜닝으로 짜낸 0.001"보다 "피처 하나의 0.01"이 크다 — 튜닝이 정체되면 피처/오류 분석으로 회귀

## 5. 시간 관리

- 튜닝은 오래 걸린다 → 셀 실행 전 예상 소요 시간을 안내하고, trial 수를 민석님이 조절할 수 있게 상수로 노출
- 중간 저장: Optuna storage(sqlite)로 study를 저장해 학교↔집 이동 시 이어서 탐색 가능하게 한다 (단, sqlite 파일은 git 제외 → 이어가려면 같은 기기이거나 study 요약을 HANDOFF에 기록)
