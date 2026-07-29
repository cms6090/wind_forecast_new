# 01_preprocessing — 원본 데이터 구조 표준화

## Why

원본 CSV는 그대로 쓸 수 없는 형태였다.
- GFS/LDAPS: `forecast_kst_dtm` 하나당 격자(grid_id)별로 여러 행이 있는 긴 형태(long format)
- SCADA: 10분 단위, 터빈별 컬럼 — 라벨(1시간, KPX 그룹별)과 시간 단위·집계 단위가 다름

이후 EDA·피처 엔지니어링·모델링이 모두 같은 기준 테이블을 쓰도록, 구조만 표준화하는 단계를 먼저 만들었다. 이 단계에서는 어떤 피처를 쓸지 고르지 않는다(그건 `03_features.ipynb`의 몫).

## How

1. **GFS/LDAPS pivot**: `forecast_kst_dtm` x `grid_id` 긴 형태를 `forecast_kst_dtm` 1행 넓은 형태로 변환. 격자는 임의로 줄이지 않고 전부(GFS 9개, LDAPS 16개) 보존했다 — 근거: 어느 격자가 터빈에 가까운지는 EDA(위경도 지도)로 확인해야 판단할 수 있어, 지금 줄이면 되돌리기 어렵기 때문(CLAUDE.md 2.5절, 근거 없는 변수 선택 금지).
2. **`data_available_kst_dtm` 규칙 검증**: data_description.md의 "01:00~익일 00:00 24시간 블록이 같은 `data_available_kst_dtm`을 갖는다"는 규칙을 실데이터로 확인. leakage-guard의 전제가 되는 규칙이라 직접 검증이 필요했다.
3. **SCADA 터빈→KPX그룹 매핑**: `info.xlsx`를 직접 읽어 확인 — VESTAS 1~6호기는 그룹1, 7~12호기는 그룹2, UNISON 1~5호기는 그룹3 (그룹설비용량 21.6/21.6/21MW과 일치).
4. **SCADA 단위·집계 방법 검증**: `_power_kw10m`을 "10분 평균출력(kW)"으로 가정하고 평균으로 집계했더니 라벨과 상관이 거의 0이 나와, 실험적으로 원인을 찾았다(근거: 실험 결과).
5. **최종 조인**: `train_base`(라벨+GFS+LDAPS+SCADA, kst_dtm 기준), `test_base`(GFS+LDAPS만, SCADA·라벨 없음 — 실제 운영에서도 test 기간엔 존재하지 않음)를 만들어 parquet으로 저장.

## Result

| 항목 | 결과 |
|---|---|
| GFS pivot | train (26304, 317), test (8760, 317) — 9격자 x 35변수 + 2 |
| LDAPS pivot | train (26304, 482), test (8760, 482) — 16격자 x 30변수 + 2 |
| `data_available_kst_dtm` 규칙 위반 | **0건** (최초 검증 코드는 달력 날짜로 묶어 1095건이 나왔으나, 이는 자정(00:00) 값이 다음날 그룹으로 새는 검증 코드 버그였다. 블록 기준으로 다시 묶어 위반 0건 확인 — 실제 데이터는 문제 없음) |
| SCADA 물리적 불가능 값 | VESTAS 868건 (전체의 0.46%, 통신 오류로 추정) 결측 처리, UNISON 0건 |
| SCADA 단위 재해석 | `_power_kw10m` = "10분간 발전량(kWh)"로 확인, 집계는 **합계**(평균 아님) |
| SCADA-라벨 상관 | kpx_group_1 = 0.9998, kpx_group_2 = 0.9998, kpx_group_3 = 0.9966 |
| train_base | (26304, 804), 결측 총 17,741건 |
| test_base | (8760, 798), 결측 총 752건 |

## So-what

- **다음 단계(`02_eda.ipynb`)는 이 parquet 캐시(`train_base.parquet`, `test_base.parquet`, `{gfs,ldaps}_grid_meta.parquet`)를 불러와 진행한다.**
- SCADA 단위 재해석(합계 vs 평균) 사례는 "컬럼명을 그대로 믿지 말고 라벨과 대조 검증한다"는 원칙을 다시 확인시켜준 사례 — 이후 GFS/LDAPS 변수 해석에도 같은 태도를 유지한다.
- `train_base`의 결측 17,741건은 대부분 예상된 결측(그룹_3 라벨이 2022년 없음, SCADA가 그룹별 시작 시점 이전 구간 없음)으로 보이지만, **컬럼별 결측 분해는 아직 안 했다** — `02_eda.ipynb`에서 확인 필요.
- `test_base`의 결측 752건은 원인이 아직 불명 — 예보 발표 누락일 후보로 보고 `02_eda.ipynb` 2절(기상 예보 커버리지 확인)에서 확인한다.
- 격자별 위경도(`gfs_grid_meta`, `ldaps_grid_meta`)는 저장만 해뒀고, 터빈과의 최근접 격자 판단은 EDA에서 지도로 확인한 뒤 `03_features.ipynb`에서 반영한다.
