"""src/measure.py — 측정·판정 계층 (재구축 v2의 토대)

이 모듈에는 **모델도 피처도 없다.** 점수를 재는 법과 판정하는 법만 있다.

왜 이걸 먼저 만드는가 — 04~05에서 내린 잘못된 결정 세 개가 전부 '모델'이 아니라
'측정'에서 나왔기 때문이다.

  1. 리더보드 오프셋이 -0.0112 ~ +0.0050으로 흔들리는데 **0.0003 차이로 순위**를 매겼다
     (v6 vs v5). 시드 σ만 0.0024다.
  2. 2024-12 group_3의 정지율이 0.815라 **B안 fold3이 오염된 걸 알면서** 동등 가중으로
     평균 냈고, 그 평균이 모든 판정의 주 지표였다.
  3. 지형 speed-up 피처를 **만든 뒤에** 효과를 쟀다. 상한 계산(학습 0회)을 먼저 했으면
     -0.096%가 나와서 반나절을 안 썼을 것이다.

그래서 이 모듈이 강제하는 것:
  · 점수는 **항상** 그룹 × fold로 분해해서 보관한다 (사후에 어떤 조합으로도 다시 합칠 수 있게)
  · 두 변형의 비교는 **fold별 짝지은 차이**로 한다 (fold 난이도 차이가 상쇄된다)
  · 피처를 만들기 전에 `headroom()`으로 **얻을 수 있는 최댓값**을 먼저 계산한다

사용:
    import src.measure as ms
    pg   = ms.per_group_scores(actual_df, pred_df)      # 그룹별 분해
    bench.add("baseline", "A안(2024)", pg)
    bench.table()                                        # 판정용 표
    bench.compare("baseline", "cand")                    # 개선 / 동률 / 악화
    ms.headroom(resid, axis)                             # 피처 만들기 전 게이트
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metric import metric, TARGET_COLS, CAPACITY_KWH

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
B_FOLDS = ["B안 fold1", "B안 fold2", "B안 fold3"]
A_FOLD = "A안(2024)"

#: 라벨 품질이 깨진 (fold, 그룹) 칸. 04의 20-2에서 발견.
#: 2024-12 group_3은 시간의 **81.5%가 1대 이상 정지**이고, 그 달이 fold3 검증구간에 들어간다.
#: "신뢰도를 낮춰 읽자"는 의도 표명만으로는 아무것도 바뀌지 않는다 — 여기 적어두고
#: `table()`이 이 칸을 뺀 '정제 B평균'을 **항상 같이** 계산하게 한다.
CONTAMINATED: set[tuple[str, str]] = {("B안 fold3", "kpx_group_3")}

#: 시드 5개 앙상블의 재현 표준편차 (04 21절 실측). 판정 문턱의 기준.
SEED_SIGMA = 0.0024

#: 판정 문턱 배수. |Δ| < K_NOISE × 짝지은 노이즈 이면 '동률'.
K_NOISE = 2.0

#: 문턱의 **하한**. 통계가 아니라 **의사결정** 기준이다.
#:
#: 왜 필요한가: fold가 3개뿐이라 fold별 Δ가 우연히 비슷하면 표준오차가 0에 가깝게
#: 붕괴하고, 그러면 0.0003짜리 차이도 "유의"로 통과한다. 실제로 자가 시험에서 그렇게 됐다.
#: (이게 정확히 v5 vs v6를 0.0003으로 순위 매긴 실패의 재현이다.)
#:
#: 왜 0.002인가:
#:   · 시드 σ 0.0024        (MLP 포함 구성의 실측 재현 산포)
#:   · B fold 간 σ 0.017    (fold 난이도 차이)
#:   · 리더보드 오프셋 변동폭 0.016  (v1~v7 실측 -0.0112 ~ +0.0050)
#:   · 1등과의 격차 0.032
#: 이보다 작은 이득은 **복잡도 비용을 못 갚는다.** 코드 한 줄로 되는 것이 아니면 채택하지 않는다.
MIN_EFFECT = 0.002


# ---------------------------------------------------------------------------
# 1. 점수를 그룹 × fold로 분해해서 보관한다
# ---------------------------------------------------------------------------
def per_group_scores(actual_df: pd.DataFrame, pred_df: pd.DataFrame,
                     row_mask=None) -> pd.DataFrame:
    """대회 산식을 **그룹별로 분해**해서 돌려준다.

    공식 `metric()`은 그룹별 nmae/ficr을 각각 단순평균한 뒤 합친다:

        총점 = 0.5·(1 − mean_g nmae_g) + 0.5·(mean_g ficr_g)

    즉 **그룹 하나가 총점에 정확히 1/3씩** 기여한다. 그룹별로 따로 들고 있으면
    나중에 어떤 부분집합으로도 다시 합칠 수 있다(오염 칸 제외 등).

    같이 담는 진단값:
      n         채점 대상 행 수 (실제발전량 >= 용량의 10%)
      bias      (예측 − 실제) 평균 ÷ 용량. 양수 = 과대예측
      sigma     (예측 − 실제) 표준편차 ÷ 용량. **FICR의 진짜 병목**
      band6/8   오차율이 밴드 안에 든 비율 (actual 가중 — FICR의 정의와 같게)

    `row_mask`: 그룹별 불리언 배열의 dict/DataFrame. 주면 **그 행만** 채점한다.
      ⚠️ 이렇게 나온 값은 **대회 점수가 아니다.** "어떤 종류의 시간대에서 지고 있나"를
         가르는 **진단용**이다(예: 전 터빈이 정상 가동 중이던 시간만 골라 채점).
    """
    rows = []
    for col in TARGET_COLS:
        a = actual_df[col].to_numpy(dtype=float)
        f = pred_df[col].to_numpy(dtype=float)
        cap = CAPACITY_KWH[col]

        valid = a >= cap * 0.10          # 산식 원문과 동일 (NaN은 False가 되어 자연 제외)
        if row_mask is not None:
            m = row_mask[col] if hasattr(row_mask, "__getitem__") else row_mask
            valid = valid & np.asarray(m, dtype=bool)
        a, f = a[valid], f[valid]
        if a.size == 0:
            rows.append(dict(group=col, nmae=np.nan, ficr=np.nan, n=0,
                             bias=np.nan, sigma=np.nan, band6=np.nan, band8=np.nan))
            continue

        err = f - a
        er = np.abs(err) / cap
        price = np.select([er <= 0.06, er <= 0.08], [4.0, 3.0], default=0.0)
        aw = a / a.sum()                 # actual 가중 (FICR과 같은 가중)

        rows.append(dict(
            group=col,
            nmae=float(er.mean()),
            ficr=float((a * price).sum() / (a * 4.0).sum()),
            n=int(valid.sum()),
            bias=float(err.mean() / cap),
            sigma=float(err.std() / cap),
            band6=float(aw[er <= 0.06].sum()),
            band8=float(aw[er <= 0.08].sum()),
        ))
    return pd.DataFrame(rows).set_index("group")


def combine(pg: pd.DataFrame, groups=None) -> tuple[float, float, float]:
    """`per_group_scores` 결과를 총점으로 되합친다. `groups`로 부분집합만 쓸 수 있다.

    ⚠️ 부분집합으로 합친 값은 **대회 점수가 아니다.** 오염 칸을 뺐을 때 결론이
       뒤집히는지 보는 **진단용**이다. 제출 판단은 전체 3그룹 값으로 한다.
    """
    d = pg if groups is None else pg.loc[[g for g in groups]]
    d = d.dropna(subset=["nmae", "ficr"])
    one_minus_nmae = 1.0 - float(d["nmae"].mean())
    ficr = float(d["ficr"].mean())
    return 0.5 * one_minus_nmae + 0.5 * ficr, one_minus_nmae, ficr


def assert_matches_official(actual_df: pd.DataFrame, pred_df: pd.DataFrame, tol=1e-12) -> None:
    """분해→재합성이 공식 `metric()`과 **비트 수준으로** 같은지 확인한다.

    측정 계층 자체에 버그가 있으면 아래 모든 판정이 무의미하므로, 실험을 시작하기 전
    한 번은 반드시 통과시킨다. (`src/metric.py`는 수정 금지 — 여기서는 읽기만 한다.)
    """
    mine = combine(per_group_scores(actual_df, pred_df))
    official = metric(actual_df[TARGET_COLS], pred_df[TARGET_COLS])
    for a, b, name in zip(mine, official, ["총점", "1-NMAE", "FICR"]):
        assert abs(a - b) < tol, f"{name} 불일치: 분해 {a!r} vs 공식 {b!r}"


# ---------------------------------------------------------------------------
# 2. 실험 저장소 — 그룹 × fold × 시드를 전부 들고 있는다
# ---------------------------------------------------------------------------
class Bench:
    """변형별 점수를 **분해된 상태 그대로** 쌓아두는 저장소.

    04~05는 fold별 총점 하나만 저장하고 평균을 냈다. 그러면 나중에
    "group_3을 빼면?", "fold3을 빼면?", "시드 산포는?"에 답하려면 전부 다시 돌려야 한다.
    (실제로 커널 재시작으로 51회 학습을 날린 적이 있다.)
    """

    def __init__(self):
        self.rows: list[dict] = []

    # -- 기록 ---------------------------------------------------------------
    def add(self, variant: str, fold: str, pg: pd.DataFrame, seed: int | None = None) -> None:
        """`per_group_scores` 결과 한 판을 기록한다."""
        for g, r in pg.iterrows():
            self.rows.append(dict(variant=variant, fold=fold, group=g, seed=seed, **r.to_dict()))

    def add_from_cache(self, ctx, pred_cache, variant: str, seed: int | None = None) -> None:
        """기존 `pl.run_variant`가 채운 `PRED_CACHE`를 그대로 흡수한다.

        기존 파이프라인을 고치지 않고도 판정 품질만 올릴 수 있게 하는 다리다.
        `pred_cache[(fold, variant, group)] = pd.Series` 규약을 그대로 따른다.
        """
        for fold, info in ctx.fold_info.items():
            key0 = (fold, variant, TARGET_COLS[0])
            if key0 not in pred_cache:
                continue
            pred = pd.DataFrame({g: pred_cache[(fold, variant, g)] for g in TARGET_COLS},
                                index=info["valid_idx"])
            self.add(variant, fold, per_group_scores(info["actual_df"], pred), seed)

    # -- 조회 ---------------------------------------------------------------
    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def fold_scores(self, variant: str, clean: bool = False) -> pd.Series:
        """fold별 총점. `clean=True`면 오염 칸(CONTAMINATED)을 뺀 값.

        시드가 여러 개면 **그룹별 지표를 시드 평균**한 뒤 합친다
        (총점을 먼저 내고 평균 내는 것과 다르다 — 산식이 비선형이므로 이쪽이 맞다).
        """
        d = self.df
        d = d[d["variant"] == variant]
        out = {}
        for fold, sub in d.groupby("fold", sort=False):
            pg = sub.groupby("group")[["nmae", "ficr"]].mean()
            groups = ([g for g in pg.index if (fold, g) not in CONTAMINATED] if clean
                      else list(pg.index))
            out[fold] = combine(pg, groups)[0] if groups else np.nan
        order = [A_FOLD] + B_FOLDS
        return pd.Series(out).reindex([f for f in order if f in out])

    def seed_sigma(self, variant: str) -> float:
        """이 변형의 실측 시드 산포(B평균 기준). 시드가 1개면 NaN."""
        d = self.df
        d = d[(d["variant"] == variant) & d["seed"].notna()]
        if d["seed"].nunique() < 2:
            return np.nan
        per_seed = []
        for _, sub in d.groupby("seed"):
            pg = sub[sub["fold"].isin(B_FOLDS)].groupby("group")[["nmae", "ficr"]].mean()
            per_seed.append(combine(pg)[0])
        return float(np.std(per_seed, ddof=1))

    def table(self) -> pd.DataFrame:
        """판정용 표.

        ⚠️ **정렬하지 않는다.** 정렬은 "1등이 있다"는 인상을 주는데, 대부분의 차이는
           노이즈 안에 있다. 순위는 `compare()`로만 매긴다.

        열 읽는 법:
          B평균        주 지표 (CLAUDE.md 5장)
          B σ          fold 간 표준편차. **이게 크면 B평균 차이를 믿지 말 것**
          B평균(정제)  2024-12 group_3 오염 칸을 뺀 값. 본 값과 방향이 다르면 채택 보류
          A안          방향 일치 확인용 보조 지표
        """
        variants = list(dict.fromkeys(self.df["variant"]))
        rows = {}
        for v in variants:
            s, sc = self.fold_scores(v), self.fold_scores(v, clean=True)
            b = s.reindex(B_FOLDS)
            rows[v] = {
                A_FOLD: s.get(A_FOLD, np.nan),
                **{f: b.get(f, np.nan) for f in B_FOLDS},
                "B평균": b.mean(),
                "B σ": b.std(ddof=1),
                "B평균(정제)": sc.reindex(B_FOLDS).mean(),
                "시드 σ": self.seed_sigma(v),
            }
        return pd.DataFrame(rows).T.round(4)

    # -- 판정 ---------------------------------------------------------------
    def compare(self, base: str, cand: str, noise: float | None = None,
                k: float = K_NOISE, verbose: bool = True) -> dict:
        """두 변형을 **fold별로 짝지어** 비교한다.

        왜 짝짓기(paired)인가: fold마다 난이도가 다르다(fold3은 원래 점수가 높다).
        평균끼리 빼면 그 난이도 차이가 노이즈로 남지만, **같은 fold끼리 먼저 빼면**
        난이도가 상쇄되고 '변형이 만든 차이'만 남는다. 같은 데이터·같은 분할을
        쓰는 실험에서는 이쪽이 항상 더 예민하다.

        채택 조건 두 개를 **모두** 만족해야 '개선'이다.
          (1) B안 3개 fold의 Δ 부호가 **전부 같다** (한 fold만 끌고 가는 개선 배제)
          (2) |평균 Δ| >= k × 노이즈

        노이즈는 두 가지 중 **큰 쪽**을 쓴다.
          · **짝지은 표준오차** = std(Δ_fold) / √3
            fold마다 효과가 얼마나 들쭉날쭉한지. 2025년으로 일반화될지를 재는 값이다.
          · **시드 노이즈** = 시드 σ × √2   (두 변형이 각각 흔들리므로 √2)
            LightGBM 기본값은 완전 결정적이라(subsample=1.0, subsample_freq=0) 0이지만,
            MLP가 섞이면 실측 시드 σ가 잡힌다.

        ⚠️ fold가 3개뿐이라 표준오차 추정 자체가 불안정하다(t분포 자유도 2).
           그래서 **부호 일치**를 같이 요구한다 — 이쪽이 더 튼튼한 기준이다.

        하나라도 어긋나면 **'동률'** 이다. 동률은 "조금 나은 것"이 아니라
        **"구별할 수 없음"** 이다 — 표에 순위를 적지 말 것.
        """
        b, c = self.fold_scores(base), self.fold_scores(cand)
        common = [f for f in b.index if f in c.index]
        diff = (c[common] - b[common])
        bdiff = diff.reindex([f for f in B_FOLDS if f in common]).dropna()

        if noise is None:
            # (a) fold 간 짝지은 차이의 표준오차 — 효과가 fold마다 얼마나 들쭉날쭉한가
            paired_se = (float(bdiff.std(ddof=1) / np.sqrt(len(bdiff)))
                         if len(bdiff) >= 2 else np.nan)
            # (b) 시드 노이즈 — 결정적 모델(LightGBM 기본값)끼리면 0이다
            sigmas = [s for s in (self.seed_sigma(base), self.seed_sigma(cand)) if np.isfinite(s)]
            seed_noise = max(sigmas) * np.sqrt(2) if sigmas else 0.0
            noise = float(np.nanmax([paired_se, seed_noise]))
        else:
            paired_se, seed_noise = np.nan, np.nan
        thresh = max(k * noise, MIN_EFFECT)      # ★ 하한 — SE 붕괴로 문턱이 0이 되는 것을 막는다

        same_sign = bool(len(bdiff) and (np.sign(bdiff) == np.sign(bdiff.iloc[0])).all()
                         and bdiff.iloc[0] != 0)
        mean_d = float(bdiff.mean())
        big = abs(mean_d) >= thresh

        if same_sign and big:
            verdict = "개선" if mean_d > 0 else "악화"
        else:
            verdict = "동률"

        # CLAUDE.md 5장: "한쪽에서만 개선되고 다른 쪽은 그대로거나 악화되면
        # '효과 불확실'로 보고 채택하지 않는다." A안이 문턱을 넘어 반대로 가면 보류한다.
        a_delta = float(diff.get(A_FOLD, np.nan))
        a_opposes = (np.isfinite(a_delta) and verdict in ("개선", "악화")
                     and np.sign(a_delta) != np.sign(mean_d) and abs(a_delta) >= thresh)
        if a_opposes:
            verdict = "효과 불확실(A안 반대)"

        # 정제 B평균도 같은 방향인지 (오염 칸이 결론을 만들고 있지 않은지)
        clean_d = float((self.fold_scores(cand, clean=True).reindex(B_FOLDS).mean()
                         - self.fold_scores(base, clean=True).reindex(B_FOLDS).mean()))
        clean_agrees = (np.sign(clean_d) == np.sign(mean_d)) or verdict == "동률"

        out = dict(verdict=verdict, mean_delta=mean_d, threshold=thresh, noise=noise,
                   paired_se=paired_se, seed_noise=seed_noise, a_delta=a_delta,
                   a_opposes=bool(a_opposes),
                   same_sign=same_sign, per_fold=diff, clean_delta=clean_d,
                   clean_agrees=bool(clean_agrees))
        if verbose:
            print(f"[{cand}] vs [{base}]  →  **{verdict}**")
            print(f"  fold별 Δ : " + "  ".join(f"{f.replace('안','')}={d:+.4f}"
                                               for f, d in diff.items()))
            src_ = "실효크기 하한" if thresh > k * noise else f"노이즈 {noise:.4f} × {k}"
            print(f"  B평균 Δ  : {mean_d:+.4f}   문턱 ±{thresh:.4f} ({src_})"
                  f"   부호일치={'O' if same_sign else 'X'}")
            if np.isfinite(paired_se):
                print(f"             노이즈 내역 — 짝지은 SE {paired_se:.4f} / 시드 {seed_noise:.4f}"
                      f" / 하한 {MIN_EFFECT:.4f}")
            print(f"  정제 Δ   : {clean_d:+.4f}  {'(방향 일치)' if clean_agrees else '⚠️ 방향 불일치 — 오염 칸이 결론을 만들고 있다'}")
            if a_opposes:
                print(f"  ⚠️ A안이 문턱을 넘어 반대 방향({a_delta:+.4f}). "
                      f"CLAUDE.md 5장 — 효과 불확실로 보고 채택하지 않는다.")
            if verdict == "동률":
                print("  ⇒ 구별 불가. 채택하지 않는다(더 단순한 쪽을 유지).")
        return out


# ---------------------------------------------------------------------------
# 3. 상한 계산기 — 피처를 만들기 **전에** 통과해야 하는 게이트
# ---------------------------------------------------------------------------
def headroom(resid, axis, bins=None, n_bins: int = 12, min_count: int = 30,
             label: str = "") -> dict:
    """어떤 축으로 조건화하면 잔차 σ를 **최대 몇 % 줄일 수 있는지**의 상한.

    쓰는 법: 새 피처가 잡으려는 물리량을 축으로 놓고 돌린다. 상한이 문턱(보통 -1%)에
    못 미치면 **그 피처는 만들지 않는다.** 학습 0회, 몇 초.

    계산 (05 2B-2c에서 확정):

        설명분산   = Σ nᵢ(mᵢ − m̄)² / N        구간 평균들이 흩어진 정도(표본가중)
        노이즈보정 = (K−1)·σ²/N                구간 K개면 백색잡음이어도 이만큼 저절로 흩어진다
        σ감소상한  = √(1 − (설명분산−노이즈보정)/σ²) − 1        (음수가 개선)

    ⚠️ **진폭(최대−최소)으로 재면 안 된다.** 05에서 이 실수로 반나절을 썼다.
       진폭은 (a) 양 극단 두 구간만 보므로 표본 적은 구간의 우연이 그대로 들어가고,
       (b) 표본이 어디 몰렸는지를 무시한다(240° 한 구간에 표본의 36%가 있는데
       표본 1개짜리 구간과 똑같이 센다).

    ⚠️ **노이즈 보정을 빼먹으면 구간을 잘게 쪼갤수록 없는 신호가 커 보인다.**
       시간대 24구간·월 12구간이 특히 위험하다.

    `min_count`: 이 미만인 구간은 버린다. 05에서 표본 2~4개짜리 꼬리가 상한을
    -0.773%까지 부풀린 적이 있다. 버린 뒤 `coverage`(남은 표본 비율)를 반드시 확인할 것 —
    0.99 미만이면 버린 구간에 진짜 신호가 있었을 수 있다.

    Returns
    -------
    dict(bound_pct, sigma, explained, noise, k_bins, n, coverage, verdict)
        `bound_pct`가 음수일수록 좋다. -1% 미만이면 만들 값어치가 있다.
    """
    r = pd.Series(np.asarray(resid, dtype=float)).reset_index(drop=True)
    x = pd.Series(np.asarray(axis)).reset_index(drop=True)
    ok = r.notna() & x.notna()
    r, x = r[ok], x[ok]
    if len(r) < 2:
        return dict(bound_pct=np.nan, sigma=np.nan, explained=np.nan, noise=np.nan,
                    k_bins=0, n=len(r), coverage=np.nan, verdict="표본 부족", label=label)

    # 구간 나누기 — 이미 범주형이면 그대로, 연속형이면 분위수로(표본을 고르게 나눈다)
    if bins is not None:
        lab = pd.cut(x, bins, include_lowest=True)
    elif pd.api.types.is_numeric_dtype(x) and x.nunique() > n_bins:
        lab = pd.qcut(x, n_bins, duplicates="drop")
    else:
        lab = x.astype("category")

    grp = r.groupby(lab, observed=True)
    cnt, mean = grp.count(), grp.mean()
    keep = cnt.index[cnt >= min_count]
    coverage = float(cnt.loc[keep].sum() / cnt.sum()) if len(keep) else 0.0
    if len(keep) < 2:
        return dict(bound_pct=np.nan, sigma=float(r.std()), explained=np.nan, noise=np.nan,
                    k_bins=len(keep), n=len(r), coverage=coverage,
                    verdict="구간 부족", label=label)

    sel = lab.isin(keep)
    rk = r[sel]
    n, k = len(rk), len(keep)
    var = float(rk.var(ddof=0))
    if var <= 0:
        return dict(bound_pct=0.0, sigma=0.0, explained=0.0, noise=0.0, k_bins=k,
                    n=n, coverage=coverage, verdict="분산 없음", label=label)

    c, m = cnt.loc[keep].to_numpy(float), mean.loc[keep].to_numpy(float)
    mbar = float((c * m).sum() / c.sum())
    explained = float((c * (m - mbar) ** 2).sum() / n)
    noise = (k - 1) * var / n
    adj = max(explained - noise, 0.0)
    bound = (np.sqrt(max(1.0 - adj / var, 0.0)) - 1.0) * 100.0

    verdict = "★ 만들 값어치 있음" if bound <= -1.0 else "기각(문턱 -1% 미달)"
    if coverage < 0.99:
        verdict += f"  ⚠️ 표본 {coverage:.1%}만 사용"
    return dict(bound_pct=float(bound), sigma=float(np.sqrt(var)), explained=explained,
                noise=float(noise), k_bins=k, n=n, coverage=coverage,
                verdict=verdict, label=label)


def group_report(bench: "Bench", variant: str, show_folds: bool = True) -> pd.DataFrame:
    """한 변형의 성적을 **그룹별로** 펼쳐서 '어느 그룹이 어디서 지는지'를 가른다.

    04는 이 갈래를 전체 평균으로만 봤다. 그런데 처방이 완전히 다르다.

      σ가 크다     → 풍속·피처. **오차 자체**를 줄여야 한다 (위치 조정은 소용없다)
      편향이 크다   → τ·라벨·가동률 복원. **위치만** 옮기면 된다 (값싸다)
      NMAE만 나쁘다 → 큰 오차(꼬리)의 문제. 손실함수·이상치

    `밴드÷σ` = 0.06 / σ. FICR의 4원 밴드(±6% 용량)가 오차 폭의 몇 배인가.
    04 전체 평균은 0.35였다(정규분포라면 통과율 27%). 그룹마다 다르면 처방도 달라야 한다.
    """
    d = bench.df
    d = d[d["variant"] == variant]
    if d.empty:
        raise KeyError(f"'{variant}' 결과가 없다. bench.add(...) 했는지 확인할 것.")
    folds = [c for c in [A_FOLD, *B_FOLDS] if c in set(d["fold"])]

    summ = d.groupby("group")[["nmae", "ficr", "sigma", "bias", "band6", "band8", "n"]].mean()
    summ.insert(0, "1-NMAE", 1 - summ["nmae"])
    summ["밴드÷σ"] = 0.06 / summ["sigma"]
    summ = summ.drop(columns="nmae")

    print(f"■ [{variant}] 그룹별 ({len(folds)} fold 평균)")
    _show(summ.round(4))

    if show_folds:
        for col, title in [("ficr", "FICR"), ("sigma", "σ (오차 폭 ÷ 용량)"),
                           ("bias", "편향 (+면 과대예측)")]:
            p = d.pivot_table(index="group", columns="fold", values=col).reindex(columns=folds)
            print(f"\n■ {title} — fold별")
            _show(p.round(4))

    # 각 그룹을 최고 그룹 수준으로 올렸을 때의 총점 이득 (그룹 하나 = 총점의 1/3)
    best = summ["ficr"].idxmax()
    print(f"\n■ FICR을 최고 그룹({best.split('_')[-1]}, {summ.loc[best,'ficr']:.4f}) 수준으로 "
          f"올렸을 때 총점 이득")
    for g, v in (0.5 * (summ.loc[best, "ficr"] - summ["ficr"]) / 3.0).sort_values(
            ascending=False).items():
        print(f"   {g:14s} FICR {summ.loc[g,'ficr']:.4f}  →  +{v:.4f}")
    print("   (참고) 04 라벨정제 +0.010 / 05 MLP블렌드 +0.010 / 1등과의 격차 0.0321")
    return summ


def _show(obj) -> None:
    """노트북이면 표로, 아니면 그냥 출력. (`display`는 IPython이 노트북 전역에만 주입한다)"""
    try:
        from IPython.display import display as _d
        _d(obj)
    except Exception:
        print(obj)


def headroom_table(resid, axes: dict, **kw) -> pd.DataFrame:
    """여러 축을 한 번에 재서 표로. `axes = {"풍향": wd, "시간대": hour, ...}`"""
    rows = [headroom(resid, v, label=k, **kw) for k, v in axes.items()]
    df = pd.DataFrame(rows).set_index("label")
    return df[["bound_pct", "sigma", "k_bins", "n", "coverage", "verdict"]] \
             .rename(columns={"bound_pct": "σ감소상한%", "sigma": "σ", "k_bins": "구간수",
                              "n": "표본", "coverage": "표본커버리지", "verdict": "판정"}) \
             .sort_values("σ감소상한%")
