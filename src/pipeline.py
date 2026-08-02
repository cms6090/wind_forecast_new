"""공통 파이프라인 — 노트북들이 import 해서 쓰는 모듈.

`04_model_selection.ipynb`의 1~21절에서 **검증을 마친 정의만** 옮겨 담았다.
노트북 셀에 흩어져 있던 500여 줄을 여기로 모은 이유는 세 가지다.

1. **커널을 새로 켤 때마다 정의 셀 15개를 순서대로 실행하는 비용**이 컸다
2. `train.ipynb` / `inference.ipynb`는 **같은 정의를 공유해야** 한다 (2차 평가 재현성 요건).
   양쪽에 코드를 복사하면 한쪽만 고치는 사고가 난다
3. 노트북이 229셀까지 길어져 읽기 어려워졌다

---

## 쓰는 법

```python
import sys; sys.path.append("..")          # notebooks/ 안에서 실행할 때
import src.pipeline as pl

ctx = pl.load_context()                     # train/test 로드 + 컬럼 분류 + fold + 가동률
X   = pl.frame_for_fold(ctx, "kpx_group_1", "2024_01")
```

## 검증 기준 (이 모듈이 옳게 옮겨졌는지 확인하는 값)

`05_pipeline.ipynb` 1절이 자동으로 확인한다.

| 구성 | A안(2024) | B안 평균 |
|---|---|---|
| `B_norm` τ=0.50 (v4 발전량 모델) | 0.6418 | 0.6356 |
| `A_asis` τ=0.60 (v2 구성) | 0.6391 | 0.6298 |

---

## ⚠️ 이 모듈이 지키는 누수 방어선

- `assert_no_leak()` — 컬럼 이름에 라벨 의존 표식(`ws_est`, `power_curve_est`)이 있으면
  **그 fold에서 허용된 2개**가 아닌 한 무조건 실패시킨다. 2026-08-01에 실제 누수를 놓쳤던
  '이름 열거' 방식 대신 **규칙 기반**으로 막는다
- 파워커브·표준화 통계·가동률은 **전부 학습 구간에서만** 적합한다
- 가동률(`ctx.avail`)은 SCADA 기반이라 test에 없다. **피처가 아니라 라벨 정제 전용**이다
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch

from .metric import metric, TARGET_COLS, CAPACITY_KWH

# ---------------------------------------------------------------------------
# 경로 · 상수
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SEED = 42
GROUP_COLS = TARGET_COLS

# 03_features와 동일한 값 (fold-safe 파생을 그 자리에서 다시 만들 때 씀)
GROUP_NEAREST_LDAPS = {"kpx_group_1": 5, "kpx_group_2": 6, "kpx_group_3": 12}
ICING_RISK_GROUPS = ["kpx_group_1", "kpx_group_2"]
CUT_IN, RATED, HIGH_WIND_THRESHOLD = 3.0, 12.0, 15.0
ICING_WS_MIN, ICING_WS_MAX = 3.0, 7.0
RHO_STANDARD = 1.225

# 검증 fold (CLAUDE.md 5장 — A안과 B안을 항상 함께 본다)
CUTOFFS = {
    "2023_07": pd.Timestamp("2023-07-01"),
    "2024_01": pd.Timestamp("2024-01-01"),
    "2024_07": pd.Timestamp("2024-07-01"),
}
END_TRAIN = pd.Timestamp("2025-01-01")
FOLD_SPECS = {
    "A안(2024)": dict(cv_suffix="2024_01", valid_start=CUTOFFS["2024_01"], valid_end=END_TRAIN),
    "B안 fold1": dict(cv_suffix="2023_07", valid_start=CUTOFFS["2023_07"], valid_end=CUTOFFS["2024_01"]),
    "B안 fold2": dict(cv_suffix="2024_01", valid_start=CUTOFFS["2024_01"], valid_end=CUTOFFS["2024_07"]),
    "B안 fold3": dict(cv_suffix="2024_07", valid_start=CUTOFFS["2024_07"], valid_end=END_TRAIN),
}
B_FOLDS = ["B안 fold1", "B안 fold2", "B안 fold3"]

# 확정된 하이퍼파라미터 (04절 실험 결과)
BEST_TOPN = 200                # 15-2: 피처 상위 200개
DEFAULT_TAU = 0.50             # 20-5 + 리더보드 v4
DEFAULT_LABEL_MODE = "B_norm"  # 20-4
WIND_TAG = "gbdtl2"
RAND_PARAMS = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)   # 15-6
DYN_SHIFTS = [1, 2, 3]

# 라벨 정제 (20절)
AVAIL_FLOOR = 0.2        # y/avail 나눗셈 폭발 방지 하한
AVAIL_NAN_FILL = 1.0     # 가동률 미상 = 정상 가동으로 간주
DOWN_WS_MIN = 5.0        # 이 풍속 이상인데
DOWN_POWER_FRAC = 0.01   # 출력이 정격의 이 비율 이하면 = 정지
TURBINES = {
    "kpx_group_1": [("vestas", i) for i in range(1, 7)],
    "kpx_group_2": [("vestas", i) for i in range(7, 13)],
    "kpx_group_3": [("unison", i) for i in range(1, 6)],
}

LABEL_DERIVED_MARKERS = ("ws_est", "power_curve_est")
WIND_EXCLUDE_MARKERS = ("ws_est", "power_curve_est", "regime_", "high_wind_caution", "icing_risk")


# ---------------------------------------------------------------------------
# 컨텍스트 — 노트북마다 전역 변수를 다시 만들지 않도록 한 덩어리로 들고 다닌다
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    train: pd.DataFrame
    test: pd.DataFrame | None
    common_cols: list                      # 그룹에 속하지 않는 공통 피처
    group_cols: dict                       # g -> 그 그룹 전용 피처 목록
    fold_info: dict                        # fold 이름 -> dict(cv_suffix, train_mask, valid_idx, actual_df)
    avail: pd.DataFrame                    # 시간별 터빈 가동률 (train 전용, 피처 아님)
    cache: dict = field(default_factory=dict)   # 풍속·프레임·중요도 캐시

    @property
    def full_mask(self):
        """fold 컷오프 없음 = 전체 train (최종 재학습용)."""
        return pd.Series(True, index=self.train.index)


def load_context(with_test: bool = True, with_avail: bool = True) -> Ctx:
    """parquet 로드 → 컬럼 분류 → fold 정의 → 가동률 계산까지 한 번에."""
    train = pd.read_parquet(PROCESSED_DIR / "train_features_v1.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_features_v1.parquet") if with_test else None

    non_feature = set([
        "kst_dtm", "data_available_kst_dtm_gfs", "data_available_kst_dtm_ldaps", "year",
        *TARGET_COLS, *[f"scada_ws_{g}" for g in GROUP_COLS], *[f"scada_{g}" for g in GROUP_COLS],
    ])
    common_cols = [c for c in train.columns
                   if c not in non_feature and not any(c.startswith(f"{g}_") for g in GROUP_COLS)]
    group_cols = {g: [c for c in train.columns if c.startswith(f"{g}_") and c not in non_feature]
                  for g in GROUP_COLS}

    fold_info = {}
    for name, spec in FOLD_SPECS.items():
        train_mask = train["kst_dtm"] < CUTOFFS[spec["cv_suffix"]]
        valid_mask = (train["kst_dtm"] >= spec["valid_start"]) & (train["kst_dtm"] < spec["valid_end"])
        valid_idx = train.index[valid_mask]
        fold_info[name] = dict(cv_suffix=spec["cv_suffix"], train_mask=train_mask,
                               valid_idx=valid_idx, actual_df=train.loc[valid_idx, TARGET_COLS])

    ctx = Ctx(train=train, test=test, common_cols=common_cols, group_cols=group_cols,
              fold_info=fold_info, avail=pd.DataFrame(index=train.index))
    if with_avail:
        ctx.avail = compute_availability(train)
    return ctx


# ---------------------------------------------------------------------------
# 터빈 가동률 (20-1절)
# ---------------------------------------------------------------------------
def compute_availability(train: pd.DataFrame) -> pd.DataFrame:
    """터빈별 SCADA로 시간별 가동률(0~1)을 만든다.

    정지 판정: 나셀 풍속 >= 5 m/s 인데 출력이 정격의 1% 이하.
    (바람이 약해서 안 도는 것은 정상이므로 풍속 조건이 필요하다.)
    1시간 집계는 01_preprocessing과 같은 `closed="right", label="right"` 규칙을 쓴다.

    ⚠️ train에만 존재한다. **피처가 아니라 라벨 정제 전용.**
    """
    src = {
        "vestas": pd.read_csv(REPO_ROOT / "data/train/scada_vestas_train.csv", parse_dates=["kst_dtm"]),
        "unison": pd.read_csv(REPO_ROOT / "data/train/scada_unison_train.csv", parse_dates=["kst_dtm"]),
    }
    hourly = {}
    for g, turbs in TURBINES.items():
        rated10 = CAPACITY_KWH[g] / len(turbs) / 6.0      # 터빈 1대의 10분 최대 발전량(kWh)
        acc = None
        for pre, i in turbs:
            d = src[pre]
            p, w = d[f"{pre}_wtg{i:02d}_power_kw10m"], d[f"{pre}_wtg{i:02d}_ws"]
            ok = 1 - ((w >= DOWN_WS_MIN) & (p <= rated10 * DOWN_POWER_FRAC)).astype(float)
            ok[w.isna() | p.isna()] = np.nan               # 계측 결측은 판정 불가
            acc = ok if acc is None else acc + ok
        s = pd.Series(acc.to_numpy(), index=src[turbs[0][0]]["kst_dtm"])
        hourly[g] = s.resample("h", closed="right", label="right").mean() / len(turbs)
    return pd.DataFrame({g: train["kst_dtm"].map(hourly[g]) for g in GROUP_COLS}, index=train.index)


# ---------------------------------------------------------------------------
# 피처 — fold-safe 파생 / 누수 점검 / 프레임 빌더
# ---------------------------------------------------------------------------
def add_fold_safe_ws_features(df, group_col, ws_col):
    """ws_col로부터 corrected/sq/cube/regime_*/high_wind_caution/icing_risk를 그 자리에서 다시 계산.
    공식·임계값은 03_features.ipynb 4/10/13절과 완전히 동일하다."""
    ws = df[ws_col]
    corrected = ws * (df[f"{group_col}_air_density"] / RHO_STANDARD) ** (1 / 3)
    out = {
        f"{ws_col}__corrected": corrected,
        f"{ws_col}__sq": corrected ** 2,
        f"{ws_col}__cube": corrected ** 3,
        f"{ws_col}__regime_calm": (ws < CUT_IN).astype(int),
        f"{ws_col}__regime_ramp": ((ws >= CUT_IN) & (ws < RATED)).astype(int),
        f"{ws_col}__regime_rated": (ws >= RATED).astype(int),
        f"{ws_col}__high_wind_caution": (ws >= HIGH_WIND_THRESHOLD).astype(int),
    }
    if group_col in ICING_RISK_GROUPS:
        tcol = f"ldaps_g{GROUP_NEAREST_LDAPS[group_col]}_heightAboveGround_2_t"
        out[f"{ws_col}__icing_risk"] = ((df[tcol] < 273.15) &
                                        ws.between(ICING_WS_MIN, ICING_WS_MAX)).astype(int)
    return pd.DataFrame(out, index=df.index)


def all_cv_cols_for_group(ctx, g):
    return [c for c in ctx.train.columns
            if c.startswith(f"{g}_ws_est_cv_") or c.startswith(f"{g}_power_curve_est_cv_")]


def leaky_cols_for_group(g):
    """fold-safe가 아닌(전체 데이터로 적합한) 라벨 의존 컬럼 — 검증에서 반드시 제외."""
    base = [f"{g}_ws_est", f"{g}_ws_est_sq", f"{g}_ws_est_cube", f"{g}_ws_est_corrected",
            f"{g}_power_curve_est",
            f"{g}_regime_calm", f"{g}_regime_ramp", f"{g}_regime_rated", f"{g}_high_wind_caution"]
    if g in ICING_RISK_GROUPS:
        base.append(f"{g}_icing_risk")
    return base


def assert_no_leak(columns, g, cv_suffix):
    """규칙 기반 누수 점검. 이름에 라벨 의존 표식이 있으면 '__' 앞부분(뿌리)을 구해서,
    그 뿌리가 이번 fold에서 허용된 2개가 아니면 무조건 실패시킨다."""
    allowed = {f"{g}_ws_est_cv_{cv_suffix}", f"{g}_power_curve_est_cv_{cv_suffix}"}
    named = set(leaky_cols_for_group(g))
    bad = [c for c in columns
           if (any(m in c for m in LABEL_DERIVED_MARKERS) and c.split("__")[0] not in allowed)
           or c in named]
    assert not bad, f"{g}/{cv_suffix} 누수 의심 컬럼 {len(set(bad))}개: {sorted(set(bad))[:8]}"


def add_forecast_dynamics(df, cols):
    """예보 계열에 시간 방향 파생(lag/lead/변화율/이동통계).
    누수 아님: 수치예보는 발표 시점에 미래 시각까지 통째로 제공되므로 lead도 합법이다."""
    dt = df["kst_dtm"]
    assert dt.is_monotonic_increasing, "kst_dtm이 시간순이 아님"
    gaps = dt.diff().dropna().unique()
    assert len(gaps) == 1 and gaps[0] == pd.Timedelta("1h"), f"1시간 간격 연속이 아님: {gaps[:5]}"

    out = {}
    for c in cols:
        s = df[c]
        for k in DYN_SHIFTS:
            out[f"{c}__lag{k}"] = s.shift(k)
            out[f"{c}__lead{k}"] = s.shift(-k)
        out[f"{c}__ramp_prev"] = s - s.shift(1)
        out[f"{c}__ramp_next"] = s.shift(-1) - s
        roll3, roll7 = s.rolling(3, center=True, min_periods=1), s.rolling(7, center=True, min_periods=1)
        out[f"{c}__roll3_mean"] = roll3.mean()
        out[f"{c}__roll7_mean"] = roll7.mean()
        out[f"{c}__roll7_std"] = roll7.std()
        out[f"{c}__roll7_max"] = roll7.max()
        out[f"{c}__roll7_min"] = roll7.min()
    res = pd.DataFrame(out, index=df.index).bfill().ffill()
    assert res.isna().sum().sum() == 0, "동적 피처에 결측 잔존"
    return res


def add_lead_features(df):
    """예보 리드타임과 raw hour. 발표 시각은 예보와 함께 오는 메타데이터라 누수가 아니다."""
    lead = (df["kst_dtm"] - df["data_available_kst_dtm_gfs"]).dt.total_seconds() / 3600.0
    return pd.DataFrame({"forecast_lead_hours": lead.astype(float),
                         "hour_raw": df["kst_dtm"].dt.hour.astype(float)}, index=df.index)


def fit_power_curve_oracle(ws, power, bin_width=0.5, min_count=10):
    """0.5 m/s 구간평균 + 단조증가 강제 (03_features 12-1절과 같은 로직)."""
    edges = np.arange(0, np.nanmax(ws) + 2 * bin_width, bin_width)
    idx = np.digitize(ws, edges) - 1
    vals = np.full(len(edges), np.nan)
    for b in range(len(edges)):
        m = idx == b
        if m.sum() >= min_count:
            vals[b] = power[m].mean()
    vals = pd.Series(vals).interpolate(limit_direction="both").fillna(0.0).to_numpy()
    return edges, np.maximum.accumulate(vals)


def apply_power_curve_oracle(ws, edges, vals):
    return vals[np.clip(np.digitize(ws, edges) - 1, 0, len(vals) - 1)]


def _assemble_frame(ctx, df, g, ws_col, pc_col, tmp, base_cols, dynamics, lead_feat):
    parts = [df[ctx.common_cols + base_cols], tmp[[ws_col, pc_col]],
             add_fold_safe_ws_features(tmp, g, ws_col)]
    if dynamics:
        parts.append(add_forecast_dynamics(tmp, [ws_col, pc_col]))
    if lead_feat:
        parts.append(add_lead_features(df))
    out = pd.concat(parts, axis=1)
    assert out.isna().sum().sum() == 0, f"{g}: 프레임에 결측"
    return out


def _wind_frame_base(ctx, df, g, ws_series, tag):
    """풍속 계열 피처를 새로 만들기 위한 공통 준비."""
    exclude = set(leaky_cols_for_group(g)) | set(all_cv_cols_for_group(ctx, g))
    base_cols = [c for c in ctx.group_cols[g] if c not in exclude]
    missing = [c for c in ctx.common_cols + base_cols if c not in df.columns]
    assert not missing, f"{g}: df에 없는 컬럼 {missing[:5]}"

    ws_col, pc_col = f"{g}_ws_{tag}", f"{g}_power_curve_{tag}"
    tmp = pd.DataFrame({"kst_dtm": df["kst_dtm"], f"{g}_air_density": df[f"{g}_air_density"]},
                       index=df.index)
    tmp[ws_col] = ws_series.astype(float)
    if g in ICING_RISK_GROUPS:
        tcol = f"ldaps_g{GROUP_NEAREST_LDAPS[g]}_heightAboveGround_2_t"
        tmp[tcol] = df[tcol]
    return base_cols, ws_col, pc_col, tmp


def build_frame_with_wind(ctx, df, g, train_mask, ws_series, tag,
                          dynamics=True, lead_feat=True, pc_target=None):
    """주어진 풍속으로 풍속 계열 피처를 전부 새로 만든 프레임 (파워커브를 '적합'한다).
    파워커브는 **학습 구간에서만** 적합한다 — 검증 구간 라벨은 절대 보지 않는다."""
    base_cols, ws_col, pc_col, tmp = _wind_frame_base(ctx, df, g, ws_series, tag)
    y_pc = df[g] if pc_target is None else pc_target
    fit_idx = train_mask & y_pc.notna()
    edges, vals = fit_power_curve_oracle(tmp.loc[fit_idx, ws_col].to_numpy(dtype=float),
                                         y_pc[fit_idx].to_numpy(dtype=float))
    tmp[pc_col] = apply_power_curve_oracle(tmp[ws_col].to_numpy(dtype=float), edges, vals)
    frame = _assemble_frame(ctx, df, g, ws_col, pc_col, tmp, base_cols, dynamics, lead_feat)
    return frame, (edges, vals)


def build_frame_given_pc(ctx, df, g, ws_series, edges, vals, tag, dynamics=True, lead_feat=True):
    """파워커브를 '적합'하지 않고 '적용'만 한다 → **라벨이 없는 test에도 쓸 수 있다.**"""
    base_cols, ws_col, pc_col, tmp = _wind_frame_base(ctx, df, g, ws_series, tag)
    tmp[pc_col] = apply_power_curve_oracle(tmp[ws_col].to_numpy(dtype=float), edges, vals)
    return _assemble_frame(ctx, df, g, ws_col, pc_col, tmp, base_cols, dynamics, lead_feat)


# ---------------------------------------------------------------------------
# 풍속 모델 (16절)
# ---------------------------------------------------------------------------
def build_wind_input_frame(ctx, df, g):
    """풍속 모델의 입력. 라벨/SCADA에서 파생된 컬럼은 전부 제외한다."""
    base = [c for c in ctx.group_cols[g] if not any(m in c for m in WIND_EXCLUDE_MARKERS)]
    dyn_src = [c for c in [f"{g}_ws10_nearest", "ldaps_ws10_avg16", "gfs_g5_ws_100m"] if c in df.columns]
    out = pd.concat([df[ctx.common_cols + base], add_forecast_dynamics(df, dyn_src),
                     add_lead_features(df)], axis=1)
    bad = [c for c in out.columns if any(m in c for m in LABEL_DERIVED_MARKERS)]
    assert not bad, f"풍속 모델 입력에 라벨 의존 컬럼 잔존: {bad[:5]}"
    assert out.isna().sum().sum() == 0, "풍속 모델 입력에 결측"
    return out


def fit_wind_gbdt(ctx, g, train_mask, objective="l2", n_estimators=3000, seed=SEED):
    """SCADA 실측 풍속을 타깃으로 LightGBM을 학습한다.
    발전량 라벨이 없어도 SCADA만 있으면 학습에 쓸 수 있다.
    ⚠️ 17-2 결과에 따라 **무작위성 없이(결정적)** 쓴다 — 풍속을 평활하면 강풍 발전량이 눌린다."""
    X = build_wind_input_frame(ctx, ctx.train, g)
    y = ctx.train[f"scada_ws_{g}"]
    fit_idx = train_mask & y.notna()
    cut = ctx.train.loc[fit_idx, "kst_dtm"].quantile(0.9)      # 뒤 10%는 early stopping용
    tr, es = fit_idx & (ctx.train["kst_dtm"] <= cut), fit_idx & (ctx.train["kst_dtm"] > cut)
    m = lgb.LGBMRegressor(objective=objective, random_state=seed, n_estimators=n_estimators,
                          verbosity=-1, importance_type="gain")
    m.fit(X.loc[tr], y[tr], eval_set=[(X.loc[es], y[es])], eval_metric="rmse",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def wind_estimate(ctx, g, cv_suffix, train_mask, objective="l2"):
    """그 fold의 학습 구간에서만 학습해 **전체 기간**의 추정 풍속을 돌려준다(fold-safe, 캐시됨)."""
    key = ("wind", g, cv_suffix, objective)
    if key not in ctx.cache:
        m = fit_wind_gbdt(ctx, g, train_mask, objective)
        X = build_wind_input_frame(ctx, ctx.train, g)
        ctx.cache[key] = pd.Series(m.predict(X), index=ctx.train.index).clip(lower=0.0)
    return ctx.cache[key]


# ---------------------------------------------------------------------------
# 라벨 처리 + LightGBM 학습 (10절 · 20절)
# ---------------------------------------------------------------------------
def make_sample_weight(y, g, mode):
    """산식을 학습에 반영하는 표본 가중.
    'actual' = 실제발전량/설비용량 (하한 0.1) — FICR의 actual 가중을 모사한다.
    하한을 두는 이유: 예측 시점엔 어떤 행이 채점 대상인지 모른다."""
    if mode is None:
        return None
    ratio = np.asarray(y, dtype=float) / CAPACITY_KWH[g]
    if mode == "eval_only":
        return np.where(ratio >= 0.10, 1.0, 0.1)
    if mode == "actual":
        return np.maximum(ratio, 0.1)
    raise ValueError(mode)


def make_label_arrays(ctx, g, idx, mode):
    """(학습 타깃, 가중치 배수, 남길 행 마스크). 20-4절 비교에서 `B_norm`이 채택됐다."""
    y = ctx.train.loc[idx, g].astype(float)
    a = ctx.avail[g].reindex(idx).fillna(AVAIL_NAN_FILL).clip(0.0, 1.0)
    if mode == "A_asis":
        return y, None, pd.Series(True, index=idx)
    if mode == "B_norm":
        return (y / np.maximum(a, AVAIL_FLOOR)).clip(upper=CAPACITY_KWH[g]), None, pd.Series(True, index=idx)
    if mode == "C_dropdown":
        return y, None, (a >= 1 - 1e-9)
    if mode == "D_weight":
        return y, np.maximum(a, 0.1) ** 2, pd.Series(True, index=idx)
    raise ValueError(mode)


def lgbm_train_label(ctx, X_full, g, train_mask, alpha=DEFAULT_TAU, seed=SEED,
                     mode=DEFAULT_LABEL_MODE, rand=False, n_estimators=2000):
    """분위수 LightGBM. 라벨/가중/행선택을 mode에 따라 바꾼다.
    ⚠️ train_mask 안에서만 동작한다 — 검증 구간 라벨·가동률은 건드리지 않는다.
    가중치는 항상 **원본 라벨**로 계산한다(산식의 actual 가중과 일치시키기 위해)."""
    train = ctx.train
    fit_idx = train_mask & train[g].notna()
    cut = train.loc[fit_idx, "kst_dtm"].quantile(0.9)
    idx_tr = train.index[fit_idx & (train["kst_dtm"] <= cut)]
    idx_es = train.index[fit_idx & (train["kst_dtm"] > cut)]

    def prep(idx):
        y_mod, mul, keep = make_label_arrays(ctx, g, idx, mode)
        w = make_sample_weight(train.loc[idx, g], g, "actual")
        if mul is not None:
            w = w * np.asarray(mul, dtype=float)
        k = keep.to_numpy()
        return X_full.loc[idx][k], y_mod[k], w[k]

    Xtr, ytr, wtr = prep(idx_tr)
    Xes, yes, wes = prep(idx_es)
    assert len(Xtr) > 100 and len(Xes) > 20, f"{g}/{mode}: 학습 표본이 너무 적음"

    params = dict(objective="quantile", alpha=alpha, random_state=seed,
                  n_estimators=n_estimators, verbosity=-1, importance_type="gain")
    if rand:
        params.update(RAND_PARAMS)
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xes, yes)], eval_sample_weight=[wes],
          eval_metric="l1", callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def select_features(model, columns, top_n=BEST_TOPN):
    """gain 중요도 상위 top_n개. top_n=None이면 gain>0인 것 전부."""
    imp = pd.Series(model.feature_importances_, index=model.feature_name_, dtype=float)
    imp = imp.reindex(columns).fillna(0.0).sort_values(ascending=False)
    return imp[imp > 0].index.tolist() if top_n is None else imp.head(top_n).index.tolist()


# ---------------------------------------------------------------------------
# fold 실험 harness
# ---------------------------------------------------------------------------
def frame_for_fold(ctx, g, cv_suffix, train_mask=None, pc_norm=False):
    """검증용 피처 프레임 (GBDT 풍속 기반, 캐시됨)."""
    key = ("frame", g, cv_suffix, pc_norm)
    if key not in ctx.cache:
        if train_mask is None:
            train_mask = next(v["train_mask"] for v in ctx.fold_info.values()
                              if v["cv_suffix"] == cv_suffix)
        ws = wind_estimate(ctx, g, cv_suffix, train_mask)
        pc_target = None
        if pc_norm:
            a = ctx.avail[g].fillna(AVAIL_NAN_FILL).clip(0.0, 1.0)
            pc_target = (ctx.train[g] / np.maximum(a, AVAIL_FLOOR)).clip(upper=CAPACITY_KWH[g])
        frame, _ = build_frame_with_wind(ctx, ctx.train, g, train_mask, ws, WIND_TAG,
                                         pc_target=pc_target)
        ctx.cache[key] = frame
    return ctx.cache[key]


def keep_for_fold(ctx, g, cv_suffix, train_mask, top_n=BEST_TOPN):
    """피처 집합을 fold마다 한 번만 고르고 고정한다(변형 간 공정 비교를 위해).
    기준은 `A_asis` τ=0.60 — 20-4절과 동일."""
    key = ("keep", g, cv_suffix, top_n)
    if key not in ctx.cache:
        X = frame_for_fold(ctx, g, cv_suffix, train_mask)
        m = lgbm_train_label(ctx, X, g, train_mask, alpha=0.60, seed=SEED, mode="A_asis")
        ctx.cache[key] = select_features(m, X.columns, top_n)
    return ctx.cache[key]


def score_predictions(actual_df, pred_df):
    """대회 공식 metric()을 그대로 사용 (src/metric.py 수정 금지 원칙)."""
    return metric(actual_df[TARGET_COLS], pred_df[TARGET_COLS])


def run_variant(ctx, pred_cache, variant, fit_fn, verbose=True):
    """fit_fn(ctx, g, cv_suffix, train_mask, valid_idx) -> pd.Series 를
    모든 fold x 그룹에 돌려 캐시에 저장하고 fold별 점수표를 돌려준다."""
    rows = []
    for fold_name, info in ctx.fold_info.items():
        preds = {}
        for g in GROUP_COLS:
            p = fit_fn(ctx, g, info["cv_suffix"], info["train_mask"], info["valid_idx"])
            pred_cache[(fold_name, variant, g)] = p
            preds[g] = p
        s, n, f = score_predictions(info["actual_df"], pd.DataFrame(preds, index=info["valid_idx"]))
        rows.append({"fold": fold_name, "model": variant, "score": s, "1-NMAE": n, "FICR": f})
        if verbose:
            print(f"  [{variant}] {fold_name}: score={s:.4f}  (1-NMAE={n:.4f}, FICR={f:.4f})")
    return pd.DataFrame(rows)


def summarize(result_dfs):
    """A안 / B안 3fold / B안 평균·표준편차 표.

    ⚠️ 지표 선택 규칙 (v5 리더보드가 알려준 것):
      · LightGBM 계열(피처·하이퍼파라미터) → **B안 평균**을 주 지표로
      · 신경망 계열(블렌드 비중·구조)     → **A안**을 주 지표로
        B안은 학습 구간이 1.5~3년이라 데이터 민감 모델을 과소평가하는데,
        최종 모델은 3년 전체를 쓴다.
    """
    df = pd.concat(result_dfs, ignore_index=True)
    p = df.pivot(index="model", columns="fold", values="score")
    p["B안 평균"] = p[B_FOLDS].mean(axis=1)
    p["B안 표준편차"] = p[B_FOLDS].std(axis=1)
    return p.round(4).sort_values("B안 평균", ascending=False)


# ---------------------------------------------------------------------------
# 산식손실 MLP — fold 단위 학습 (21절)
# ---------------------------------------------------------------------------
#: 기본 설정. ⚠️ hidden/dropout/lr/patience는 **이전 프로젝트가 179피처로 정한 값**이며
#: 우리 데이터로 검증한 적이 없다(`T_SOFT`만 21-3에서 스윕했다). 05 노트북 4절에서 튜닝한다.
MLP_CFG = dict(t_soft=0.006, hidden=256, p_drop=0.15, lr=1e-3, weight_decay=1e-4,
               max_epochs=400, patience=60, eval_every=5, top_n=BEST_TOPN,
               ficr_weight=1.0)   # λ>1 = "NMAE를 조금 내주고 밴드 안으로" (src/nn.py 설명 참조)


def _cfg_key(cfg):
    return tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in cfg.items()))


def metric_mlp_predict(ctx, g, cv_suffix, train_mask, cfg=None, seed=SEED, verbose=False):
    """산식손실 MLP를 그 fold의 학습 구간에서만 학습하고, 전체 기간 **이용률** 예측을 돌려준다.

    LightGBM과 **같은 피처 프레임**을 쓴다(공정 비교). 다른 것은 모델과 손실뿐이다.
    ⚠️ 라벨은 **원본**을 쓴다. 산식 손실은 실제 산식을 최적화하므로 `B_norm`으로 정규화하면
       존재하지 않는 타깃의 밴드를 맞추게 되어 실제 채점과 어긋난다.
    ⚠️ 학습 행은 **채점 대상만**(이용률 >= 0.10). 산식의 정의역이 거기다.
    """
    from . import nn as mnn                      # 순환 import 방지를 위해 지역에서

    cfg = {**MLP_CFG, **(cfg or {})}
    key = ("mlp", g, cv_suffix, _cfg_key(cfg), seed)
    if key in ctx.cache:
        return ctx.cache[key]

    cap = CAPACITY_KWH[g]
    X = frame_for_fold(ctx, g, cv_suffix, train_mask)
    top_n = cfg["top_n"]
    keep = list(X.columns) if top_n == "all" else keep_for_fold(ctx, g, cv_suffix, train_mask, top_n)
    Xv = X[keep].to_numpy(dtype=np.float64)
    ratio = (ctx.train[g] / cap).to_numpy(dtype=float)

    fit_idx = (train_mask & ctx.train[g].notna()).to_numpy() & (ratio >= mnn.EVAL_MIN_RATIO)
    times = ctx.train["kst_dtm"].to_numpy()
    cut = pd.Series(ctx.train.loc[fit_idx, "kst_dtm"]).quantile(0.9)
    tr = fit_idx & (times <= np.datetime64(cut))      # 앞 90% = 학습
    es = fit_idx & (times > np.datetime64(cut))       # 뒤 10% = 조기 종료용

    mu, sd = mnn.fit_standardizer(Xv[tr])             # 표준화는 학습 구간에서만
    Xes_t = torch.tensor(((Xv[es] - mu) / sd).astype(np.float32))
    y_es = ratio[es]

    def eval_fn(model):
        with torch.no_grad():
            return mnn.group_score(y_es, np.clip(model(Xes_t).numpy(), 0.0, 1.0))  # ★ 대회 산식

    model, best_ep = mnn.train_metric_mlp(
        (Xv[tr] - mu) / sd, ratio[tr], seed=seed, n_epochs=cfg["max_epochs"],
        t_soft=cfg["t_soft"], eval_fn=eval_fn, hidden=cfg["hidden"], p_drop=cfg["p_drop"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        patience=cfg["patience"], eval_every=cfg["eval_every"],
        ficr_weight=cfg["ficr_weight"])
    pred = mnn.predict_ratio(model, Xv, mu, sd)
    if verbose:
        print(f"    mlp/{g}/{cv_suffix} seed={seed}: {best_ep}/{cfg['max_epochs']}에폭, "
              f"피처 {len(keep)}개, 학습 {tr.sum()}행")
    ctx.cache[key] = (pred, best_ep)
    return ctx.cache[key]


def metric_mlp_fit_fn(cfg=None, seeds=(SEED,), verbose=False):
    """run_variant에 넘길 fit_fn. seeds가 여러 개면 **이용률 예측을 평균**한다.
    (17절의 '볼록 변환 전 평균 금지'는 풍속 얘기다. 여기는 파이프라인 마지막 단이라 해당 없음.)"""
    def fit_fn(ctx, g, cv_suffix, train_mask, valid_idx):
        preds = [metric_mlp_predict(ctx, g, cv_suffix, train_mask, cfg, s, verbose)[0] for s in seeds]
        out = pd.Series(np.mean(preds, axis=0) * CAPACITY_KWH[g], index=ctx.train.index)
        return out.loc[valid_idx].clip(lower=0, upper=CAPACITY_KWH[g])
    return fit_fn


def blend_score(ctx, pred_cache, var_a, var_b, w):
    """캐시된 두 변형을 (1-w)·A + w·B 로 섞어 fold별 점수를 낸다 (재학습 없음)."""
    rows = []
    for fold_name, info in ctx.fold_info.items():
        preds = {g: (1 - w) * pred_cache[(fold_name, var_a, g)] + w * pred_cache[(fold_name, var_b, g)]
                 for g in GROUP_COLS}
        s, n, f = score_predictions(info["actual_df"], pd.DataFrame(preds, index=info["valid_idx"]))
        rows.append({"fold": fold_name, "score": s, "1-NMAE": n, "FICR": f})
    d = pd.DataFrame(rows).set_index("fold")
    return d["score"]["A안(2024)"], d["score"][B_FOLDS].mean(), d["score"][B_FOLDS].min(), d


def lgbm_fold_fit_fn(mode=DEFAULT_LABEL_MODE, tau=DEFAULT_TAU, seed=SEED, top_n=BEST_TOPN):
    """run_variant에 넘길 LightGBM fit_fn 생성기."""
    def fit_fn(ctx, g, cv_suffix, train_mask, valid_idx):
        X = frame_for_fold(ctx, g, cv_suffix, train_mask)
        keep = keep_for_fold(ctx, g, cv_suffix, train_mask, top_n)
        m = lgbm_train_label(ctx, X[keep], g, train_mask, tau, seed, mode)
        p = pd.Series(m.predict(X.loc[valid_idx, keep]), index=valid_idx)
        return p.clip(lower=0, upper=CAPACITY_KWH[g])
    return fit_fn
