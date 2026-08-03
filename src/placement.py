"""src/placement.py — 밴드 안 배치 (행마다 예측을 어디에 놓을지 따로 정한다)

## 왜 이게 남은 유일한 카드인가

06_rebuild 4층까지의 결론: **오차(σ)를 줄이는 길은 전부 막혔다.**
고장 보정(2~6%), group_3 따라잡기, 데이터 늘리기, 2022년 빼기, 바람 조건별 보정 9축(05),
바람 공간정보 3축(4층) — 전부 문턱 미달로 닫혔다.

그런데 점수의 절반인 FICR은 **계단**이다.

    오차율 ≤ 6% → 4원 / ≤ 8% → 3원 / 넘으면 0원

**5.9%나 0.1%나 똑같이 4원**이다. 이미 밴드 안에 든 예측을 더 정밀하게 만들어봐야 한 푼도
안 는다. 반대로 밴드 밖 예측을 안으로 밀어넣으면 4원이 통째로 들어온다.

즉 오차를 못 줄여도 **"어디에 놓느냐"만 바꿔서 점수를 올릴 수 있다.**

## 04가 닫은 것과 무엇이 다른가

04는 이 문제를 **τ(분위수) 하나를 모든 행에 똑같이 적용**하는 방식으로 근사했고,
"위치 최적화는 끝났다"고 결론 냈다. v3(τ=0.60)에서 편향을 +0.023 밀었는데 FICR이
0.0008밖에 안 움직였고 1-NMAE만 0.0075 잃었다.

**그건 '모든 행을 같은 방향으로 미는 방식'이 끝났다는 뜻이다.** 여기서 하는 것은 다르다.

  · 정격 근처 시간: 실제값이 설비용량 쪽에 몰려 있다(위가 막혀 있으므로) → 위로 놓는 게 유리
  · 중간 풍속 시간: 좌우로 넓게 퍼져 있다 → 가운데가 유리

**한 행에서 유리한 방향이 다른 행에서는 불리하다.** 전역 τ는 이 둘을 구분할 수 없다.

## 어떻게 푸나

점수를 한 행씩 미분해서 보면, 행 i의 기여는 이렇다.

    기여 = −(0.5/N)·E[채점여부 · |f−a|/용량]  +  (0.125/A)·E[채점여부 · a · 단가(f,a)]
           └───────── NMAE 항 ─────────┘        └────────── FICR 항 ──────────┘

    N = 채점 대상 행 수,  A = 채점 대상 행의 실제발전량 합  (둘 다 전체 상수)

`a`(실제 발전량)는 모르지만 **분위수 모델 여러 개로 그 분포를 추정할 수 있다.**
그러면 위 식을 f에 대해 수치적으로 최대화하면 된다. 행마다 따로.

⚠️ **분포 추정을 그대로 믿으면 안 된다.** 우리 분위수 모델은 표본가중(actual/용량, 하한 0.1)을
걸고 학습해서 **가중 분위수**를 추정한다. 실제 관측 비율과 어긋난다(06 5층 검산에서 확인).
그래서 폭(`spread`)과 위치(`shift`)를 보정하는 손잡이 두 개를 두고,
**과거 fold에서 맞춘 뒤 미래 fold에 적용**한다(누수 방지).
"""
from __future__ import annotations

import numpy as np

BAND4, BAND3 = 0.06, 0.08          # 4원 / 3원 밴드 (오차율 기준, 산식 원문)
PRICE4, PRICE3 = 4.0, 3.0
EVAL_RATIO = 0.10                  # 실제발전량이 용량의 이 비율 이상인 행만 채점


# ---------------------------------------------------------------------------
# 1. 분위수 예측 -> 실제값 분포의 대표 표본
# ---------------------------------------------------------------------------
def cdf_samples(q, taus, n_samples: int = 41, spread: float = 1.0,
                shift: float = 0.0, cap: float | None = None) -> np.ndarray:
    """분위수 예측 몇 개로 '실제값이 어디쯤 나올지'를 등확률 표본으로 편다.

    Parameters
    ----------
    q : (행, 분위수개수) 분위수 예측. 행 안에서 뒤집혀 있어도(교차) 정렬해서 쓴다
    taus : q의 각 열에 대응하는 분위수 수준 (예: [0.1, 0.25, 0.5, 0.75, 0.9])
    spread : 폭 배율. **1보다 크면 분포를 넓게** 본다 (중앙값 기준으로 벌린다)
    shift : 용량 대비 이동량. 분포 전체를 위/아래로 민다
    cap : 주면 [0, cap]으로 자른다 (발전량은 음수일 수 없고 용량을 못 넘는다)

    Returns
    -------
    (행, n_samples) — 각 열이 같은 확률을 갖는 대표값. 기댓값은 그냥 행방향 평균으로 구한다.
    """
    q = np.sort(np.asarray(q, dtype=float), axis=1)
    taus = np.asarray(taus, dtype=float)
    assert q.shape[1] == taus.size >= 3, "분위수는 3개 이상 필요하다"

    i50 = int(np.argmin(np.abs(taus - 0.5)))
    med = q[:, [i50]]
    q = med + spread * (q - med) + (shift * (cap if cap else 1.0))

    # 양 끝을 한 칸씩 연장해 0%와 100% 지점을 만든다 (선형 외삽)
    lo = q[:, [0]] - (q[:, [1]] - q[:, [0]])
    hi = q[:, [-1]] + (q[:, [-1]] - q[:, [-2]])
    xs = np.concatenate([lo, q, hi], axis=1)                 # (행, T+2)
    ts = np.concatenate([[0.0], taus, [1.0]])                # (T+2,) 모든 행 공통

    p = (np.arange(n_samples) + 0.5) / n_samples             # 등확률 지점
    j = np.clip(np.searchsorted(ts, p, side="right") - 1, 0, ts.size - 2)
    w = (p - ts[j]) / (ts[j + 1] - ts[j])
    out = xs[:, j] * (1.0 - w) + xs[:, j + 1] * w
    return np.clip(out, 0.0, cap) if cap is not None else out


# ---------------------------------------------------------------------------
# 2. 행마다 기대 점수를 최대로 만드는 위치 찾기
# ---------------------------------------------------------------------------
def _unit_price(err_ratio: np.ndarray) -> np.ndarray:
    return np.where(err_ratio <= BAND4, PRICE4,
                    np.where(err_ratio <= BAND3, PRICE3, 0.0))


def best_placement(samples: np.ndarray, cap: float, w_nmae: float, w_ficr: float,
                   n_grid: int = 161, chunk: int = 2000) -> np.ndarray:
    """행마다 기대 기여를 최대로 만드는 예측값을 찾는다.

    `samples` : `cdf_samples`가 돌려준 (행, K) 등확률 표본
    `w_nmae`  = 0.5 / N      (N = 채점 대상 행 수)
    `w_ficr`  = 0.125 / A    (A = 채점 대상 행의 실제발전량 합)
                 └ 0.5(산식의 FICR 비중) × 1/4(최대단가) = 0.125

    후보 위치는 그 행의 분포가 퍼져 있는 범위 안에서 `n_grid`개를 고른다.
    메모리 때문에 행을 `chunk`개씩 끊어 처리한다.
    """
    n = samples.shape[0]
    out = np.empty(n, dtype=float)
    g = np.linspace(0.0, 1.0, n_grid)[None, :]

    for s0 in range(0, n, chunk):
        s = samples[s0:s0 + chunk]                            # (m, K)
        lo, hi = s.min(1, keepdims=True), s.max(1, keepdims=True)
        F = lo + (hi - lo) * g                                # (m, G) 후보 위치
        scored = (s >= EVAL_RATIO * cap)                      # (m, K) 채점되는 표본인가

        er = np.abs(F[:, :, None] - s[:, None, :]) / cap      # (m, G, K)
        sc = scored[:, None, :]
        nmae_term = (er * sc).mean(axis=2)                    # (m, G)
        ficr_term = (_unit_price(er) * s[:, None, :] * sc).mean(axis=2)

        obj = -w_nmae * nmae_term + w_ficr * ficr_term
        out[s0:s0 + chunk] = F[np.arange(F.shape[0]), obj.argmax(axis=1)]
    return out


def objective_weights(actual_train: np.ndarray, cap: float) -> tuple[float, float]:
    """학습 구간 라벨로 N(채점 행 수)과 A(채점 행 발전량 합)를 재서 가중 두 개를 만든다.

    ⚠️ 검증 구간 라벨을 쓰면 누수다. 반드시 **학습 구간**만 넘길 것.
    """
    a = np.asarray(actual_train, dtype=float)
    a = a[np.isfinite(a)]
    sc = a[a >= EVAL_RATIO * cap]
    if sc.size == 0:
        raise ValueError("채점 대상 행이 없다")
    return 0.5 / sc.size, 0.125 / sc.sum()
