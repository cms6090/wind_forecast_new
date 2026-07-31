# HANDOFF — 마지막 갱신: 2026-07-31 (장소: 학교, 세션 2회차 종료 — 집에서 이어감)

## 새 세션 시작 시 가장 먼저 할 것
1. 이 파일 전체를 읽는다 (특히 "이번까지 발견된 버그·주의사항" — 같은 실수를 반복하지 않기 위해 필수)
2. `git log --oneline -5`, `git status`로 커밋 상태 확인 — **학교에서 push했는지 반드시 확인**(아래 "환경 메모" 참고). push 안 됐으면 집 컴퓨터엔 `04_model_selection.ipynb`가 아예 없을 수 있음
3. `reports/02_eda.md`, `reports/03_features.md`를 훑어 결론을 확인한다 (이 문서의 요약보다 원문이 항상 더 정확함 — 요약과 다르면 리포트를 우선한다)
4. 새 기기(집)라면 `venv` 새로 만들고 `pip install lightgbm xgboost catboost scikit-learn` + `pip install torch --index-url https://download.pytorch.org/whl/cpu` 필요(아래 "환경 메모" 참고, `requirements.txt` 아직 없음), `data/`와 `data/processed/` 캐시도 없으면 복사하거나 `01_preprocessing.ipynb`부터 재생성 필요
5. 민석님에게 "`04_model_selection.ipynb`(52셀, 1~9절)에서 7절 베이스라인 사다리 4종은 민석님이 학교에서 직접 실행해 AI 사전 검증값과 정확히 일치 확인했습니다. 9절(후보 모델 비교: LightGBM/XGBoost/CatBoost 각각 기본값+단조제약, MLP)은 코드는 다 있지만 학교에서 실행 중간에 멈추고 집에서 마저 돌리기로 하셨습니다. 커널을 새로 시작해서 9절부터(또는 처음부터) 실행해주시면 결과 보고 이어가겠습니다" 정도로 보고 후 시작

## 현재 위치
- 로드맵 단계: 6. 피처 엔지니어링 완료 → **7. 모델 선택 진행 중** (`model-selection` 스킬)
- 작업 중 파일: `notebooks/04_model_selection.ipynb` (**52개 셀, 1~9절 + 10절 자리표시자**)
  - 1절 셋업, 2절 데이터 로드, 3절 모델 입력 피처 정의(공통/그룹전용/제외), 4절 A안·B안 fold 정의, 5절 **fold-safe 파생 피처 재계산**, 6절 피처 프레임 빌더+채점 함수, **7절 베이스라인 사다리 4종 — 민석님이 학교에서 직접 실행 완료, AI 사전 검증값과 정확히 일치 확인됨(아래 표)**, 8절 요약(베이스라인용, 9절 추가 전 작성분이라 다소 낡음 — 10절이 최신 요약)
  - **9절(신규, 이번 세션에 코드만 작성 — 아직 실행 미완료)**: 후보 모델 비교 — LightGBM/XGBoost/CatBoost 각각 기본값 + 단조제약(monotonic constraint) 버전, 그리고 MLP(PyTorch). 민석님이 학교에서 실행을 시작했으나(9-4절의 큰 루프, `candidate_results`) **중간에(A안 fold, group_1 처리 중 추정) 멈추고 "집에 가서 할게"로 결정** — 집에서 커널 재시작 후 이어서 실행 필요
  - 10절: "요약 및 다음 단계" 자리표시자 마크다운만 있음(9절 결과 나오면 채울 예정)
  - **주의**: 9-4절 셀(`candidate_results` 루프)에 학교에서 중간까지 실행되다 만 지저분한 출력(LightGBM deprecation 경고만 있고 실제 결과는 없음)이 남아있을 수 있음. 집에서 다시 실행하면 덮어써지므로 무시하고 그냥 재실행하면 됨.
- `reports/04_model_selection.md`는 **아직 작성 안 함** — 9절 실행 확인 후 작성 예정
- `03_features.ipynb`(86셀, 1~13절)은 지난 세션에 완료, `reports/03_features.md`도 완료 상태 유지
- EDA(`02_eda.ipynb` 1~6절)는 완료 상태 유지, `reports/02_eda.md`도 완료
- (참고) `03_features.ipynb` 여러 셀에서 `PerformanceWarning: DataFrame is highly fragmented` 경고가 뜨지만 에러 아님 — 무시해도 됨(우선순위 낮은 정리 대상)

## 7절 베이스라인 결과 — 민석님 실행으로 확정됨 (AI 사전 검증값과 일치)

| baseline | A안(2024) | B안 fold1 | B안 fold2 | B안 fold3 |
|---|---|---|---|---|
| 0_시간x월평균(기상 미사용) | 0.4334 | 0.4188 | 0.4486 | 0.4183 |
| 1_물리파워커브(ML 없음) | 0.5825 | 0.5550 | 0.5653 | 0.6006 |
| 2_선형회귀(피처 8개) | 0.5797 | 0.5429 | 0.5614 | 0.6001 |
| 3_LightGBM기본값(876피처) | 0.5965 | 0.5697 | 0.5947 | 0.6154 |

**중요한 패턴**: 0<1<3, 2<1(모든 fold에서 선형회귀가 물리 파워커브보다 근소하게 낮음)은 버그가 아니라 **파워커브의 S자형 비선형(컷인/정격 꺾임)을 3차 다항식이 잘 못 따라가서** 생기는, 도메인적으로 설명 가능한 결과. LightGBM이 1·2를 모두 확실히 이겨서(모든 fold) 트리 기반 비선형 모델이 이 문제에 적합하다는 것도 확인됨. 자세한 해석은 노트북 7-5절 마크다운 참고.

## 지난 세션(들)에서 한 것 — 누적 요약
- `01_preprocessing.ipynb`: GFS/LDAPS pivot, SCADA 10분→1시간 집계, train_base/test_base parquet 캐시 생성 (완료, `reports/01_preprocessing.md`)
- `02_eda.ipynb` 1~6절: 타깃 분포·기상예보·타깃×기상 심화·SCADA 심화·시간 무결성·DS/도메인 후속검증 — EDA 완료 (완료, `reports/02_eda.md`)
- `03_features.ipynb` (이번 세션 전체, 1~13절 작성 완료):
  - 1절: 라벨 결측(짧은 것만 보간, 4일짜리 긴 구간은 그대로 NaN 유지) / test LDAPS 결측 3시점 ffill / **curtailment 확정 3구간을 학습 라벨에서 제외**
  - 2절(+2-1 근거검증): 그룹별 최근접·16격자평균 LDAPS 풍속, GFS 4개 높이(g5) 풍속
  - **3절(+3-2b/3-2c 근거검증, 핵심): SCADA 회귀 기반 추정풍속(`{group}_ws_est`), Ridge로 다중공선성 해결** — 아래 "결정 사항"에 상세
  - 3-5절: fold-safe `ws_est_cv_*` 버전 (fold-purity 해결)
  - 4절(+4-1 근거검증): 공기밀도 보정 + 효과 검증(효과 미미함을 솔직히 확인)
  - 5절: v², v³ (파워커브 동력학)
  - 6절: 풍향 sin/cos
  - 7절: 시간(hour/month) sin/cos
  - 9절: GFS-LDAPS 앙상블 차이 피처
  - 10절: group_1/2 전용 결빙(icing) 위험 피처 — **검증 시 SCADA로 통제해야 하는 함정을 겪고 고침**
  - 11절: LDAPS 50m max/min 기반 돌풍성(gust) 피처
  - **12절(신규): 파워커브 변환 피처(`{group}_power_curve_est`) — fold-safe 버전 포함, 이중 관점 재검토에서 나온 최우선 과제였음**
  - **12-4절(신규, 코드 없는 마크다운): 12절 재검토로 발견한 잔여 리스크 3가지 기록(아래 "결정 사항"·"미해결 질문 6번" 참고)**
  - **13절(신규): 파워커브 구간 더미(calm/ramp/rated) + 고풍속 주의 피처 — 원래 계획한 "컷아웃 구간/근접 위험"은 데이터 근거상 의미 없어서 기준을 정직하게 수정함**
  - 8절: `train_features_v1.parquet`(26304,888)/`test_features_v1.parquet`(8760,860) 저장
- **`04_model_selection.ipynb`(이번 세션, 1~8절 39셀 신규 생성)**:
  - 3절: 모델 입력 피처를 공통(804개)/그룹전용(그룹당 23~24개)/제외(10개)로 분류
  - 4절: A안(2024)·B안 3-fold의 학습/검증 구간 경계(`CUTOFFS`, `FOLD_SPECS`) 정의. A안과 B안 fold2는 학습 구간이 완전히 같아서(2022~2023 전체) 같은 `_cv_2024_01` 컬럼을 재사용
  - **5절(핵심 신규 발견): `ws_est` 파생 6종(sq/cube/corrected/regime_calm·ramp·rated/high_wind_caution/icing_risk)이 전부 fold-safe하지 않은 `ws_est`에서 나온 것을 발견 → 각 fold의 `ws_est_cv_*`로부터 그 자리에서 재계산하는 `add_fold_safe_ws_features()` 작성** (민석님이 "폴드별 즉석 재계산" 방식으로 승인) — 아래 "결정 사항" 참고
  - 6절: `build_group_feature_frame()`(그룹×fold별 안전한 피처 프레임 생성)과 `score_predictions()`(공식 metric 래핑)
  - 7절: 베이스라인 사다리 4종(시간×월평균/물리파워커브/선형회귀/LightGBM기본값) — AI가 사전 검증한 결과는 위 "이번 세션 결과 미리보기" 참고
  - venv에 `lightgbm 4.7.0`, `xgboost 3.3.0`, `catboost 1.2.10`, `scikit-learn 1.9.0` 신규 설치

## 다음 할 일 (우선순위순)
0. **가장 먼저: `04_model_selection.ipynb` 9절(9-4절 `candidate_results` 루프)을 집에서 커널부터 새로 시작해 실행.** 7종(lightgbm/lightgbm_mono/xgboost/xgboost_mono/catboost/catboost_mono/mlp) × A안·B안 3-fold × 3그룹 = 84번 학습이라 시간이 꽤 걸릴 수 있음(학교에서 CatBoost가 특히 느린 걸 확인함 — 아래 "버그·주의사항" 12번 참고). 끝나면 `pivot_candidate` 표를 그대로 공유
1. 9절 결과 나오면 **세 가지를 확인**(9-4절 마지막 마크다운에 이미 질문 형태로 적어둠): ① 어떤 GBDT 라이브러리가 제일 좋은가 ② 단조 제약이 A안+B안 모두에서 개선되는가(한쪽만 개선되면 "효과 불확실") ③ MLP가 최고 GBDT를 이기는가(못 이기면 지금 단계에선 후보 제외, 앙상블 단계에서만 재고려)
2. 확인되면 10절(요약)을 채우고, 그다음 실험으로 **그룹별 3모델 vs 통합 1모델 구조**, **타깃 스케일(kWh vs 이용률)** 실험 추가 — `model-selection` 스킬 4절 참고
3. `{group}_high_wind_caution`(특히 group_1은 표본 0.36%로 매우 적음)이 실제로 유효한 피처인지 9절 LightGBM feature importance로 확인
4. 9절까지 확인 후 `reports/04_model_selection.md`를 Why/How/Result/So-what 구조로 작성

## 결정 사항 / 근거 (누적)

### 이번 세션에서 새로 정한 것 (03_features.ipynb)

- **주력 풍속 피처가 "허브고도 외삽"에서 "SCADA 회귀 추정풍속"으로 바뀜.** 경위(민석님이 낸 아이디어이자 판단이므로 상세히 기록):
  1. **1차 시도(기각)**: 윈드시어 멱법칙으로 GFS 10m/100m에서 시어지수 α를 구해 LDAPS 10m을 117m로 외삽. 발전량 상관이 오히려 나빠짐(group_1 0.73→0.60, 저풍속 구간 log비율 불안정).
  2. **분석 중 발견**: "최근접 LDAPS 격자 1개" vs "16격자 평균" 중 뭐가 나은지 그룹마다 정반대(group_1은 평균 유리 +0.11~0.12, group_2/3은 최근접 유리 -0.02~-0.05). 3년 내내 일관, Steiger's Z 검정(p<0.001)으로 우연 아님 확인.
  3. **2차 시도(채택, 민석님 제안)**: SCADA를 정답 삼아 6개 예보 풍속을 회귀로 결합. 2022~2023 학습 → 2024 out-of-sample 검증(curtailment 제외 반영):

     | | 기존 최고 단일 예보 피처 | 회귀 추정풍속 |
     |---|---|---|
     | group_1 | 0.793 | **0.854** |
     | group_2 | 0.814 | **0.855** |
     | group_3 | 0.827 | **0.869** |

     학습/검증 성능 차이 거의 없음(과적합 징후 없음). **채택.**
  4. **통계적 유의성까지 코드로 검증**: 2-1(최근접 vs 평균), 3-2b(회귀 vs 기존최고) 둘 다 Steiger's Z로 p<0.001, |Z|>20 확인. 노트북에 재현 가능한 코드로 남김.
  5. **fold-purity 문제 발견 및 해결**(민석님이 "fold-purity가 뭐냐"→"바로 고쳐"): 전체 데이터로 학습한 `ws_est`를 A안/B안 검증에 그대로 쓰면 검증 구간이 이미 정답을 참고한 값이라 불공정. **3-5절에서 학습구간 경계(2023-07-01/2024-01-01/2024-07-01)별 fold-safe 버전 3종(`ws_est_cv_2023_07/2024_01/2024_07`) 추가**. fold-safe A안(2024) 상관이 원래 out-of-sample과 사실상 동일(0.85~0.87대) — 성능 손실 없음 확인.
  6. **회귀계수 불안정(다중공선성) 발견 및 Ridge로 해결**(민석님이 이중 관점 재검토에서 지적): GFS 80m/100m 계수가 그룹별로 -2.5~1.8까지 요동(입력 변수 다수가 서로 강상관). **3-2c에서 Ridge(`RIDGE_ALPHA=30`)로 전환** — out-of-sample 성능 유지·개선하면서(group_2 0.8553→0.8556, group_3 0.8690→0.8697) 계수 절댓값 최댓값 뚜렷이 감소(예: group_2 2.25→1.53). 이후 3-3/3-5/12절 전부 Ridge 사용.

- **공기밀도 보정 효과, 검증해보니 미미함(민석님 지적, DS 관점)**: `ws_est_corrected`(밀도보정)가 `ws_est`(원본)보다 실제로 나은지 상관계수로 직접 확인. 결과: group_1 +0.0013, group_2 +0.0011, group_3 -0.0006 — **사실상 잡음 수준, 뚜렷한 개선 없음**. `02_eda`에서 이미 "SCADA 실측 풍속 쓰면 계절/밀도 효과 대부분 사라짐"을 확인했었는데 `ws_est`가 그 효과를 이미 흡수해서로 추정. 컬럼은 유지하되(도메인 근거는 유효, 비선형 모델에서 다르게 쓰일 수 있음) "효과 불확실한 후보"로 정직하게 남김.

- **파워커브 변환 피처 추가(민석님 지적, 도메인 관점 최우선 과제였음)**: `wind-domain-features` 스킬 6절 권장대로, SCADA(풍속,발전량) 곡선을 0.5m/s 구간+단조증가 강제로 만들고 `ws_est`를 통과시켜 `{group}_power_curve_est` 생성(12절). `ws_est`보다 뚜렷이 개선(out-of-sample 2024: 0.854→0.859, 0.856→0.871, 0.870→0.878). **이 피처는 SCADA뿐 아니라 라벨(발전량)까지 직접 참고하므로 fold-purity 위험이 `ws_est`보다 크다 — 처음부터 fold-safe 버전(`power_curve_est_cv_*`, 3-5와 동일한 `CV_CUTOFFS` 재사용)을 같이 만듦.**

- **12절 잔여 리스크 3가지를 12-4절에 기록(민석님이 12절 재설명 요청 후 이중 관점 재검토에서 발견, 지금 당장 고치지 않고 기록만)**:
  1. (DS) 고풍속(15m/s+) 구간은 표본이 적은데(train 0.36~1.74%), `fit_power_curve`의 `np.maximum.accumulate`(누적최댓값 단조증가 강제)는 특정 bin의 우연한 튄 평균값을 그 이후 모든 bin에 영구 전파함 — 정격구간 전체 과대추정 위험.
  2. (DS) `ws_est`는 SCADA와 완벽 일치 아닌 추정치(상관 0.83~0.87)인데, 파워커브가 가장 가파른 ramp 구간(컷인 3m/s~정격 12m/s)에서는 풍속오차가 발전량오차로 증폭됨 — 아직 regime별 잔차로 확인 안 함.
  3. (도메인) `np.maximum.accumulate`로 만든 곡선은 절대 안 떨어짐 — 실제 터빈의 컷아웃(25m/s 근처 안전정지, 페더링) 급락을 반영 못함. 지금까지 3년 train/test엔 25m/s 이상이 없었지만 2025년 태풍(8~9월) 시에는 있을 수 있음.
  - 두 DS 리스크와 도메인 리스크 3번은 결국 같은 취약 구간(고풍속)을 가리킴. 평가 산식이 설비용량 10% 미만은 채점 제외라 영향은 제한적일 가능성 높으나, `04_model_selection`에서 `regime_calm/ramp/rated`별 잔차를 확인해 필요하면 클리핑/평활화를 검토.

- **파워커브 구간 더미 + 고풍속 주의 피처 추가, 단 기준을 정직하게 수정함(민석님 지적, 도메인 관점)**: 원래 계획은 calm/ramp/rated/**cutout** 4구간 + 컷아웃(25m/s) 근접 위험 플래그였음. 그런데 `ws_est`(회귀로 여러 예보를 blend한 추정치)의 실제 분포를 확인하니 **train 3년 전체에서 25m/s를 넘는 시각이 단 한 번도 없음**(회귀가 여러 예보를 평균 내면서 극값이 눌리기 때문). 그 기준 그대로 쓰면 항상 0인 무의미한 컬럼이 되므로: **구간 더미는 calm/ramp/rated 3개만 만들고**(13-1), **고풍속 플래그는 실제로 신호가 있는 15m/s 기준의 "고풍속 주의(`high_wind_caution`)"로 이름·기준을 바꿈**(13-2, train 기준 0.31~1.74%). "컷아웃 근접"이라는 원래 이름은 오해를 부를 수 있어 쓰지 않음.

- **curtailment 확정 구간(2023-02-13~17, 2024-01-18~24, 2024-02-22~03-01)을 학습 라벨에서 제외**. 판단 근거: 평가 산식이 발전량 10% 미만은 채점 안 하므로, curtailment 구간(발전량 0)을 학습에서 빼는 건 leakage가 아니라 노이즈 제거. GFS/LDAPS 입력은 그대로 두고 라벨만 NaN 처리.
- 라벨 결측 처리: `kpx_group_1/2`의 2022-10-24~27 나흘 구간은 그대로 NaN 유지(보간 안 함). 6시간 이내 짧은 결측만 선형보간.
- test_base LDAPS 결측 3시점은 ffill로 처리.
- **결빙(icing) 위험 피처(group_1/2 전용)**: 판단 기준은 영하(0℃ 미만) AND 3~7m/s. 이 범위 자체도 SCADA 풍속구간별 감소율표로 근거를 남김(하한 3은 컷인 근방+절대발전량 무시가능 수준이라 잡음, 상한 7은 group_1엔 맞지만 group_2는 9~10m/s까지 잔여효과 있어 "근사치"임을 인정). **검증은 반드시 SCADA 실측 풍속 구간으로 통제**(ws_est로 통제하면 반대 결과 나오는 함정 있었음).
- **GFS-LDAPS 앙상블 차이**: `{group}_gfs_ldaps_diff`(평균 -2.26~-2.68, GFS가 체계적으로 낮음), `_absdiff`.
- **돌풍성(gust) 피처**: `{group}_gust_proxy`, LDAPS 50m u/v 최대-최소 범위의 벡터 크기. 발전량과 단독 상관 약함(-0.02~-0.05)이 의도된 결과.

### 이번 세션에서 새로 정한 것 (04_model_selection.ipynb)

- **`ws_est` 파생 6종의 fold-safe 처리 방식 발견 및 결정**: `03_features.ipynb`를 다시 확인하다가, `{group}_ws_est_sq/_cube/_corrected`(4절), `{group}_regime_calm/ramp/rated`(13-1절), `{group}_high_wind_caution`(13-2절), `{group}_icing_risk`(10절) 이 6종이 전부 fold-safe하지 않은 `{group}_ws_est`(2022~2024 전체 데이터로 학습한 회귀)를 그대로 통과시켜 만들어졌다는 걸 발견함(`{group}_ws_est` 본체와 `{group}_power_curve_est`만 `_cv_2023_07/2024_01/2024_07` fold-safe 버전이 있었음). 예측기준시점 원칙(leakage-guard)상, A/B 검증 fold에 이 6종을 그대로 쓰면 회귀계수를 통해 미래 정보가 은근히 섞여 들어감(3-2c에서 Ridge 계수가 fold별로 안정적이었다는 걸 이미 확인했으니 누수 강도 자체는 크지 않을 가능성이 높지만, "크지 않을 것 같다"는 추정만으로 넘어가는 건 6.2절 원칙 위반).
  - **민석님께 두 가지 방안을 제시하고 선택받음**: (a) 폴드별로 `ws_est_cv_*`를 입력으로 같은 공식(고정 임계값 3.0/12.0/15.0m/s)을 그 자리에서 재계산 vs (b) A/B 검증에서는 이 6종을 아예 빼고 최종 재학습에만 포함. **민석님이 (a) 선택** — 새 fit이 필요 없이 산술 재계산만 하면 되고, HANDOFF 미해결 질문 4번(`high_wind_caution` 효과 확인)도 제대로 답할 수 있다는 이유.
  - `04_model_selection.ipynb` 5절 `add_fold_safe_ws_features()`로 구현. `sq/cube`는 `corrected` 기준(03_features 3절 순서와 동일 — corrected를 먼저 만들고 그 값을 제곱/세제곱), `regime_*`/`high_wind_caution`/`icing_risk`는 원본 `ws`(보정 전) 기준 — 03_features 원본 코드 순서를 그대로 반영.
  - **`{group}_air_density`는 재계산 안 함**: LDAPS 기온·기압만으로 계산돼 라벨/SCADA 의존이 없어 애초에 안 leaky함. fold별로 그대로 재사용.

- **베이스라인 사다리에서 선형회귀(#2)가 물리 파워커브(#1)보다 모든 fold에서 낮게 나옴 — 버그 아니라 도메인적으로 설명됨**: `model-selection` 스킬은 "사다리가 아래에서 위로 올라간다"고 가정하지만, 실제로는 0<1<3이고 2(선형회귀, 추정풍속+제곱+세제곱)는 1(물리 파워커브)보다 근소하게 낮았다. 파워커브는 컷인(3m/s)~정격(12m/s)에서 급격히 꺾이고 그 이후 평평해지는 S자형 비선형인데, 3차 다항식은 이 꺾임(특히 정격 이후 평평한 부분)을 잘 못 따라가기 때문. 물리 파워커브(SCADA 실측을 0.5m/s 구간별 평균)는 이 비선형을 있는 그대로 담고 있어 더 정확함. LightGBM(#3)은 1·2를 모두 확실히 이겨서(모든 fold) 트리 기반 비선형 모델이 이 문제에 적합하다는 것도 같이 확인됨 — "누수 의심 신호"가 아니라 "다항식 회귀의 한계를 보여주는 정상적 결과"로 판단. **민석님이 이 설명을 듣고 "그럼 파워커브처럼 만들 수는 없냐, MLP는?"라고 제안 → 9절로 이어짐(아래).**

- **9절에 단조 제약(monotonic constraint)과 MLP를 함께 포함하기로 결정(민석님 제안, 둘 다 채택)**: 민석님이 "평가기준을 저것처럼(파워커브처럼) 만들 수 없냐, MLP를 쓰면 안 되냐"고 질문 → 두 가지 다 도메인 지식과 `model-selection` 스킬 원칙에 부합하는 제안이라 9절에 반영:
  - **단조 제약**: "바람이 세지면 예측 발전량도 커지거나 그대로여야 한다"는 파워커브의 물리적 성질을 트리 모델에 직접 강제. `{group}_ws_est_cv_*`, `{group}_power_curve_est_cv_*` 두 컬럼에만 +1(비감소) 제약을 걺(다른 피처는 단조성 근거 없어 제약 안 함). LightGBM/XGBoost/CatBoost 전부 `monotone_constraints` 파라미터로 지원.
  - **MLP**: `model-selection` 스킬 원칙("딥러닝은 GBDT 대비 명확한 개선을 보일 때만 도입")에 따라, 실제로 GBDT를 이기는지 9절에서 같이 비교해본 뒤 채택 여부 결정. `CLAUDE.md` 폴더 구조에 이미 `src/nn.py`(LightGBM과 블렌드되는 MLP, 앙상블 단계용)가 계획돼 있었는데, 그 아이디어를 지금 시점에 앞당겨 검증하는 것.
  - venv에 **PyTorch 2.13.0+cpu**(GPU 버전 아님 — 2차 평가 재현성 위해 특정 GPU 환경 의존 없이 어디서나 돌아가는 CPU 버전 선택) 신규 설치.

- **LightGBM의 `monotone_constraints`가 `regression_l1`(MAE)/`quantile` 목적함수와 호환 안 됨을 발견·수정**: 직접 돌려보니 `LightGBMError: Cannot use monotone_constraints in regression_l1 objective`. LightGBM 문서 확인 결과, 단조 제약 알고리즘이 2차 도함수를 쓰는데 L1/quantile 손실은 2차 도함수가 0이라 애초에 라이브러리 차원에서 막아둔 것. **해결: LightGBM 단조 제약 버전만 손실을 `huber`로 바꿈**(`model-selection` 스킬의 대체 손실 목록에 있는 선택지 — MAE와 비슷하게 이상치에 강건하면서 2차 도함수가 있어 제약과 호환됨). XGBoost·CatBoost는 이 제약이 없어 `reg:absoluteerror`/`MAE` 그대로 사용. early stopping 판정 기준(`eval_metric`)은 네 조합 모두 MAE로 통일해 공정성 유지.

### 02_eda 최종본에서 이어받은 것 (요약, 원본은 `reports/02_eda.md`)
- 검증 전략: A안(2022~2023 학습→2024 검증) + B안(확장 윈도우 3-fold) 항상 병행
- `kpx_group_3`은 2022년 라벨 자체가 없음
- 격자 매핑: GFS 전 그룹 공통 g5, LDAPS는 group_1→g5, group_2→g6, group_3→g12
- 후류효과 관측됐으나 그룹 단위 모델링이라 추가 조치 없음
- 예보 오차 하한: LDAPS-SCADA RMSE 2~3m/s — 모델 튜닝으로 못 줄이는 구조적 한계
- 풍향 피처는 그룹별로 설계(3개 그룹 다 WSW/W가 주풍향, group_3만 W가 크게 앞섬)

## 노트북 재사용 가능한 이름·규칙 — 다음 노트북 작성 시 참고
- `wind_speed(df, u, v)` = `sqrt(u**2+v**2)`, `wind_direction_deg(df, u, v)` = `(270-degrees(atan2(v,u)))%360` — `02_eda`/`03_features` 공통
- `GROUP_NEAREST_LDAPS = {"kpx_group_1": 5, "kpx_group_2": 6, "kpx_group_3": 12}`, `GFS_NEAREST_GRID = 5`
- `TURBINE_GROUP_MAP`: VESTAS 1~6→group_1, 7~12→group_2, UNISON 1~5→group_3
- SCADA 그룹 집계: 10분→1시간 **평균**(발전량과 달리 합계 아님), `resample("h", closed="right", label="right")`
- `steiger_z_dependent(r_a, r_b, r_ab, n)`: 종속 상관계수 차이 유의성 검정 — 2-1에서 정의, 3-2b에서 재사용. `scipy.stats` 필요
- `fit_ridge(X, y, alpha)`: 능형회귀(numpy만으로, 절편은 규제 안 함) — 3-2-code에서 정의, `RIDGE_ALPHA=30`(3-2c에서 확정) 사용. `fit_ols`도 비교용으로 남아있음
- `fit_power_curve(scada_ws, power, bin_edges)` / `apply_power_curve(ws, bc, vals)`: 파워커브(구간평균+단조증가 강제) 적합/적용 — 12-1에서 정의
- `CV_CUTOFFS = {"2023_07":..., "2024_01":..., "2024_07":...}`: fold-safe 재학습용 컷오프, 3-5에서 정의, 12-3에서 재사용(같은 패턴을 SCADA/라벨 참고 피처에 재사용 가능)
- `03_features.ipynb` 컬럼 네이밍: `{group}_ws10_nearest`, `ldaps_ws10_avg16`, `gfs_g5_ws_{10m,80m,100m,850hPa}`, `scada_ws_{group}`(회귀 학습용, train 전용), **`{group}_ws_est`(Ridge, 전체데이터학습 — 최종 재학습 전용)**, **`{group}_ws_est_cv_{2023_07,2024_01,2024_07}`(fold-safe — 검증용)**, `{group}_ws_est_corrected`(밀도보정, 효과 불확실), `{group}_ws_est_sq/cube`, `{group}_wd_sin/cos`, `hour_sin/cos`, `month_sin/cos`, `{group}_gfs_ldaps_diff/absdiff`(9절), `{group}_icing_risk`(10절, group_1/2만), `{group}_gust_proxy`(11절), **`{group}_power_curve_est`(+`_cv_*`, 12절 — 발전량 프록시, 최우선 후보)**, **`{group}_regime_calm/ramp/rated`(13-1)**, **`{group}_high_wind_caution`(13-2, 15m/s 기준)**
- **`train_features_v1.parquet`에는 절대 모델 입력으로 쓰면 안 되는 컬럼이 섞여 있음**: `scada_ws_kpx_group_*`, `scada_kpx_group_*`(01_preprocessing 산출), `year`(검증용 임시) — test_features_v1.parquet에는 없으므로 `04_model_selection`에서 피처 목록 고를 때 반드시 제외
- `04_model_selection.ipynb`에서 정의한 재사용 가능한 함수: `add_fold_safe_ws_features(df, group_col, ws_col)`(5절), `build_group_feature_frame(df, g, cv_suffix)`(6절, cv_suffix=None이면 최종 재학습용 원본 컬럼 사용), `score_predictions(actual_df, pred_df)`(6절, `src/metric.metric()` 래핑), `build_monotone_spec(columns, g, cv_suffix)`(9-1절), `fit_predict_gbdt(model_type, g, cv_suffix, train_mask, valid_idx, monotone)`(9-2절, model_type∈{lightgbm,xgboost,catboost}), `SimpleMLP`/`fit_predict_mlp(...)`(9-3절, PyTorch). `FOLD_SPECS`/`CUTOFFS`(4절)는 A안·B안 fold 경계 딕셔너리로 이후 노트북(05_tuning 등)에서도 그대로 재사용 가능.

## 이번까지 발견된 버그·주의사항 (같은 실수 반복 방지용 — 꼭 읽을 것)
1. **연속-0 구간(zero-run) 탐지 로직 함정**: `run_id = (~is_zero).cumsum()`로 그룹을 나누면, 각 그룹은 "0이 아닌 값 1개 + 뒤따르는 0들"로 구성된다. `is_zero.loc[grp.index].all()`로 검사하면 **거의 항상 False**. 올바른 방법: `zero_s = s[is_zero]; zero_run_id = run_id[is_zero]` 로 0인 값만 먼저 걸러낸 뒤 groupby.
2. **노트북 파일이 커지면 `Read`/`NotebookEdit` 도구가 실패한다**: 별도 `.py` 스크립트로 `json.load`/`json.dump`를 써서 노트북을 직접 조작한다. `python -c "..."`로 bash에 백틱 포함 텍스트를 직접 넣지 말 것(명령어 치환 오작동). `03_features.ipynb`도 이 방식으로 계속 확장했다(43→48→58→62→85셀).
3. **`NotebookEdit`으로 여러 셀 순서대로 삽입 시**: `edit_mode="insert"`는 anchor 바로 뒤에 넣으므로 여러 개면 거꾸로 반복 삽입. json 스크립트 방식은 `cells[idx:idx] = [...]`로 원하는 위치에 바로 삽입 가능해 더 편함.
4. **`01_preprocessing.ipynb`에서 겪은 실수**: SCADA `_power_kw10m`을 "10분 평균출력"으로 가정해 평균 집계했더니 라벨과 상관 거의 0 — 실제로는 "10분간 발전량(kWh)"이라 **합계**가 맞았다. 컬럼명을 그대로 믿지 말고 항상 라벨/실측과 대조 검증.
5. **시각화는 "대체"가 아니라 "추가"가 기본**: boxplot을 히스토그램으로 통째로 바꿨다가 "둘 다 하라고" 피드백 받음.
6. **AI가 물리 공식으로 만든 파생 피처는 실제로 검증하기 전엔 신뢰하지 말 것**: 시어지수 외삽이 오히려 원래 피처보다 나빴던 사례. 새 피처는 반드시 raw 대비 상관(가능하면 out-of-sample)을 확인하고 채택 여부를 결정.
7. **추정치로 만든 피처를 "다른 추정치 구간"으로 검증하면 착시가 생긴다**: 결빙 위험 피처를 `ws_est` 구간으로 통제하면 반대 결과가 나옴 — SCADA(실측)로 통제해야 정상적으로 나옴. 통제 변수 자체가 오차 있는 추정치면 가능한 한 실측값으로 통제할 것.
8. **구간·임계값을 정할 때 "EDA에서 확인했다"고만 쓰고 실제 숫자를 안 보여주면 안 된다**: 결빙 위험의 "3~7m/s"를 처음엔 근거 숫자 없이 썼다가 지적받음 — 나중에 보니 상한(7)이 group_2엔 정확히 안 맞는 근사치였음. 모든 임계값은 경계 근처 실제 수치(구간별 표)를 같이 남길 것.
9. **라벨/SCADA(정답)를 참고해서 만드는 피처는 처음부터 fold-safe하게 설계해야 한다**: `ws_est`를 fold-purity 없이 만들었다가 나중에(3-5) 따로 고쳤는데, 다음에 만든 파워커브 변환 피처는 처음부터 `CV_CUTOFFS` 패턴을 같이 적용해서 이 실수를 반복하지 않았다.
10. **"물리적으로 그럴듯한 임계값(예: 컷아웃 25m/s)"도 실제 피처 분포를 찍어보기 전엔 쓰지 말 것**: 파워커브 구간 더미·고풍속 플래그를 처음 설계할 때 "컷아웃 25m/s"를 그대로 쓰려 했는데, `ws_est`의 실제 분포(회귀로 여러 예보를 blend해서 극값이 눌림)를 확인하니 train·test 모두 25m/s를 한 번도 안 넘어서 **완전히 상수인(항상 0) 컬럼이 될 뻔했다.** 새 범주형/이진 피처를 만들 때는 반드시 그 기준으로 실제 몇 건이 걸리는지(0건이면 무의미) 먼저 세어보고 임계값을 정할 것 — 8번 교훈(임계값은 숫자로 확인)의 연장선.
11. **fold-safe 피처 하나(`ws_est`)를 만들었다고 그로부터 파생된 모든 컬럼이 자동으로 fold-safe인 게 아니다**: `03_features.ipynb`에서 `ws_est`/`power_curve_est`는 fold-safe `_cv_*` 버전을 만들어놓고, 정작 그 `ws_est`에서 파생된 sq/cube/corrected/regime_*/high_wind_caution/icing_risk 6종은 그대로 원본 `ws_est`를 참조하게 놔뒀다(9번 교훈 "처음부터 fold-safe하게 설계"를 지켰다고 생각했는데, 파생의 파생까지는 못 챙긴 사례). 새 피처가 leaky한 컬럼을 입력으로 쓰면, 그 새 피처도 자동으로 leaky해진다 — 피처 하나를 fold-safe하게 고쳤으면 그로부터 파생된 다른 모든 컬럼까지 연쇄적으로 점검할 것.
12. **AI가 사전검증용으로 백그라운드에서 돌린 스크립트가, 민석님이 동시에 노트북을 직접 실행 중이던 것과 CPU를 놓고 경합해서 둘 다 느려짐**: 9절 코드를 다 쓴 뒤 AI가 확인 삼아 같은 로직을 백그라운드로 돌렸는데, 알고 보니 민석님도 그 시점에 노트북(9절 포함)을 학교에서 직접 실행하고 있었음(Jupyter 커널 프로세스가 CPU를 크게 쓰고 있는 걸 뒤늦게 발견). 같은 무거운 연산(GBDT+MLP 84번 학습)을 두 프로세스가 동시에 돌리니 서로 느려지기만 하고 의미가 없었음 — **AI 프로세스를 즉시 종료**. 교훈: 무거운 자체 검증을 백그라운드로 돌리기 전에 "지금 민석님이 노트북을 직접 실행 중인지"부터 확인(또는 물어보기)할 것. 특히 노트북이 이미 열려 있고 무거운 셀(9절 같은)을 막 추가한 직후라면 더더욱 미리 확인.
13. **CatBoost 기본 설정(`n_estimators=2000`, GPU 미사용)이 LightGBM/XGBoost보다 훨씬 느림**: 9절 첫 실행(학교)에서 group_1 하나 처리하는 데도 꽤 오래 걸림 — CatBoost의 ordered boosting(기본 알고리즘)이 원인일 가능성 높음. 84번(7모델×4fold×3그룹) 전체를 돌리면 상당히 오래 걸릴 수 있으니, 집에서 실행할 때 너무 오래 걸리면 `n_estimators` 상한을 낮추거나 `task_type`/`thread_count` 조정을 고려. 지금은 일단 그대로 두고 인내심 있게 기다려보는 쪽으로 진행 중.

## 실험 기록
| 날짜 | 실험 | 로컬 Score | 리더보드 | 결론 |
|---|---|---|---|---|
| (아직 없음 — 베이스라인 이전 단계) | | | | |

## 미해결 질문
1. group_1/2의 2022-10-24~27 결측 나흘 구간, 지금은 그대로 NaN 유지(보간 안 함)로 확정했지만 실제 모델 성능에 영향 있는지는 아직 안 봄
2. group_3(UNISON)만 결빙 효과가 없는 이유(터빈 제작사 차이 추정)는 공식 스펙 문서 없이는 확정 불가 — 그래서 group_3엔 결빙 피처를 아예 안 만들었음
3. 후류효과의 정량적 반영 — 그룹 단위 모델링 구조상 "안 함"으로 결론냈으나, 모델 성능이 기대에 못 미치면 재검토 후보
4. `{group}_high_wind_caution`이 실제로 모델 성능에 도움이 되는지 — 표본이 매우 적은 그룹(group_1 0.36%)도 있어 `04_model_selection`에서 feature importance로 확인 필요
5. 풍향×풍속(주풍향 정렬도) 상호작용 피처는 아직 미구현 — 우선순위 상대적으로 낮음(파워커브 변환·구간 더미로 이미 상당 부분 커버될 가능성)
6. **12절 파워커브 피처의 잔여 리스크 3가지(위 "결정 사항" 참고) — 아직 실제로 문제인지 확인 안 함**. 트레이드오프: (a) 지금 바로 클리핑/평활화로 손보면 안전하지만 근거 없이 손대는 것(6.2절 "그냥 좋을 것 같다" 금지 원칙 위반 위험) / (b) `04_model_selection`에서 regime별 잔차를 먼저 보고 실제로 고풍속 구간 오차가 크게 나오는지 확인한 뒤 필요할 때만 고치는 것(현재 선택, 근거 기반 원칙에 부합) — (b)로 진행하되, 확인을 잊으면 안 됨

## 환경 메모
- **scipy**(지난 세션), **lightgbm 4.7.0 / xgboost 3.3.0 / catboost 1.2.10 / scikit-learn 1.9.0**(이번 세션 초반), **torch 2.13.0+cpu**(이번 세션 후반, MLP용, CPU 전용 빌드로 설치 — GPU 의존성 없어 2차 평가 재현성에 유리)를 venv에 설치함. **`requirements.txt`는 아직 없음** — 모델 선택·튜닝 단계 패키지가 다 들어간 뒤 한꺼번에 정리하기로 결정(2026-07-31). 새 기기(집)에서는 `pip install lightgbm xgboost catboost scikit-learn`과 `pip install torch --index-url https://download.pytorch.org/whl/cpu`를 따로 실행해야 함
- **⚠️ 커밋·push 상태 반드시 확인**: 마지막으로 확인된 원격 동기화 지점은 `d79ba1a`(03_features 완료 시점)이다. **이번 세션에 새로 만든 `notebooks/04_model_selection.ipynb`(52셀)와 이 HANDOFF.md 갱신본은 세션 종료 시점까지 커밋 여부 미확정** — AI가 커밋을 제안했고 민석님이 직접 `git add/commit/push`를 실행해야 하므로, **집에 가기 전 반드시 push까지 완료했는지 확인**. 안 했으면 집 컴�터에서 `git pull` 해도 오늘 작업이 없다.
- **`03_features.ipynb` 1~13절 전체를 민석님이 직접 실행 완료함(에러 0건, AI 사전 검증 수치와 완전히 일치).** `data/processed/train_features_v1.parquet`/`test_features_v1.parquet`도 민석님 실행으로 최종 확정됨.
- **`04_model_selection.ipynb` 7절(베이스라인)은 민석님이 학교에서 직접 실행 완료, AI 사전 검증 수치와 정확히 일치 확인됨.** 9절(후보 모델 비교)은 코드까지만 작성됐고, 학교에서 실행을 시작했으나 중간에 멈추고 **집에서 이어서 실행하기로 함** — 아직 확정된 결과 없음.
- `02_eda.ipynb`는 파일이 커서(122셀) 도구로 편집할 때 "버그·주의사항 2번" 방식을 써야 한다. `04_model_selection.ipynb`는 52셀로 커져서 이제 이 방식(json 스크립트)을 계속 써야 한다.
