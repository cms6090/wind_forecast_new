"""
대회 산식(Score)을 **직접 손실함수로 삼아** 학습하는 신경망(MLP).

`04_model_selection.ipynb` 21절이 이 모듈을 import 하고,
앞으로 만들 `train.ipynb`/`inference.ipynb`도 **같은 정의를 import** 한다.
(노트북마다 신경망 정의를 복사해 두면 한쪽만 고치는 사고가 난다.)

---

## 왜 이 손실인가

우리 프로젝트는 13~20절 내내 **트레이드오프 곡선 위를 미끄러지기만** 했다.
τ를 올리면 FICR이 오르고 1-NMAE가 깎이고, 내리면 반대였다.
20절의 리더보드 실패(v3)가 그 한계를 못 박았다 —
**편향을 +0.023 더 밀었는데 FICR은 0.0008만 움직이고 1-NMAE만 0.0075 잃었다.**

곡선을 **바깥으로** 미는 방법은 하나뿐이다: **목표(손실)를 산식 자체로 바꾸는 것.**

## 산식을 손실로 (핵심 아이디어)

한 그룹의 점수(이용률 단위, 즉 발전량 ÷ 설비용량):

    score = 0.5*(1 - mean|ŷ - y|) + 0.5 * Σ y·p(e) / (4 Σ y),   e = |ŷ - y|
    p(e)  = 4 (e<=0.06), 3 (e<=0.08), 0 (그 밖)      <- 계단 함수라 미분 불가

계단 p를 시그모이드 두 개로 매끄럽게 바꾼다:

    p_soft(e) = 3·σ((0.08 - e)/T) + σ((0.06 - e)/T)

    e ≪ 0.06     -> 3 + 1 = 4  ✔
    0.06 < e < 0.08 -> 3 + 0 = 3  ✔
    e > 0.08     -> 0 + 0 = 0  ✔

이제 `-score`를 손실로 쓰면 경사하강법이 **대회 점수 자체를 최대화**한다.

**이 손실이 하는 일**: 어떤 시각이 6% 밴드에 들어갈 가망이 있으면 그쪽으로 강하게 당기고,
가망이 없으면 그냥 L1처럼 다룬다. **시각마다 다르게 행동한다.**
분위수 τ를 올려 **모든 시각을 똑같이** 위로 미는 방식과 근본적으로 다르다.
20절에서 확인한 "τ로는 더 못 짜낸다"는 벽을 이 구조가 통과한다.

## ⚠️ 왜 LightGBM으로는 안 되는가 (시도하지 말 것)

부스팅은 잎 값을 `-Σg/Σh`로 정한다. 비볼록한 밴드 보너스가 만든 기울기를 그 규칙이
제대로 처리하지 못한다. (LightGBM 내장 `l1`은 잎 값을 잔차의 중앙값으로 다시 계산하는
특수 보정 `RenewTreeOutput`이 있지만 커스텀 목적함수에는 그 보정이 없다.)
밴드 항을 완전히 끄고 매끄러운 L1만 돌려도 내장 `l1`보다 나빴다.
**경사하강법에는 그런 제약이 없다. 그래서 신경망이다.**

## ⚠️ 왜 full-batch인가

`metric_loss`의 FICR 항은 **전체 합의 비율**(`Σ y·p / 4Σ y`)이다.
미니배치가 작으면 이 비율의 추정이 흔들려 기울기가 잡음투성이가 된다.
표본 1.5만 × 피처 200이면 CPU full-batch로 충분하다.

## ⚠️ 학습 행은 '채점 대상'만

대회는 **실제 발전량이 설비용량의 10% 이상인 시각만** 채점한다.
산식의 정의역이 거기이므로 학습도 거기서 한다.
(LightGBM 쪽은 `actual/capacity` 표본가중 하한 0.1로 같은 취지를 근사했다.)
"""

import os
import random

import numpy as np
import torch
import torch.nn as tnn

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
T_SOFT = 0.006          # 계단을 부드럽게 하는 폭. 21-4절에서 우리 피처로 재검증한다
HIDDEN = 256            # 은닉층 폭 (표본 1.5만에 맞는 작은 모델)
DROPOUT = 0.15
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MAX_EPOCHS = 400
EVAL_EVERY = 5          # 몇 에폭마다 검증 산식을 볼지
PATIENCE = 60           # 이만큼 개선이 없으면 조기 종료

# FICR 밴드 경계 — src/metric.py의 값과 반드시 같아야 한다
BAND_FULL = 0.06        # 이 이내면 단가 4 (만점)
BAND_PART = 0.08        # 이 이내면 단가 3
EVAL_MIN_RATIO = 0.10   # 채점 대상 기준: 실제 발전량 >= 설비용량 x 10%


def set_seed(seed: int, deterministic: bool = True) -> None:
    """파이썬·numpy·torch 난수를 모두 고정한다 (2차 평가 재현성 요건).

    deterministic=True면 비결정적 커널 사용을 막아 비트 단위 재현이 된다.
    이 프로젝트는 CPU만 쓰므로(모델이 작다) 이것만으로 충분하다.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


class MetricMLP(tnn.Module):
    """n_in -> 256 -> 256 -> 1. 출력은 sigmoid로 [0, 1](이용률)에 가둔다.

    sigmoid로 가두는 이유: 발전량은 0 미만이나 설비용량 초과가 물리적으로 불가능하다.
    애초에 그 범위 밖을 예측하지 못하게 하면 학습이 쉬워진다.
    (LightGBM은 이 제약을 못 걸어 나중에 clip으로 잘라내야 했다.)

    입력: (배치, n_in) float32. **반드시 표준화된 값**(standardize 참조).
    출력: (배치,) 이용률 예측 [0, 1]
    """

    def __init__(self, n_in: int, hidden=HIDDEN, p_drop: float = DROPOUT):
        """hidden은 int(같은 폭 2층) 또는 tuple(층마다 폭 지정) 둘 다 받는다."""
        super().__init__()
        widths = (hidden, hidden) if isinstance(hidden, int) else tuple(hidden)
        layers, prev = [], n_in
        for h in widths:
            layers += [tnn.Linear(prev, h), tnn.BatchNorm1d(h), tnn.GELU(), tnn.Dropout(p_drop)]
            prev = h
        layers += [tnn.Linear(prev, 1)]
        self.net = tnn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def soft_price(e: torch.Tensor, t_soft: float = T_SOFT) -> torch.Tensor:
    """대회 단가 계단함수(4 / 3 / 0)의 미분 가능한 근사.

    입력: e — 오차율 |ŷ - y| (이용률 단위 = 발전량 오차 ÷ 설비용량)
    출력: 단가 근사값 (약 0 ~ 4)
    T가 작을수록 실제 계단에 가깝지만 기울기가 날카로워 학습이 불안정해진다.
    """
    return 3.0 * torch.sigmoid((BAND_PART - e) / t_soft) + torch.sigmoid((BAND_FULL - e) / t_soft)


def metric_loss(pred: torch.Tensor, y: torch.Tensor, t_soft: float = T_SOFT,
                ficr_weight: float = 1.0) -> torch.Tensor:
    """대회 산식을 그대로 옮긴 손실 (최소화 대상 = -score).

        L = 0.5 * mean|ŷ - y|  -  0.5 * λ * Σ y·p_soft(e) / (4 Σ y)
            └─ NMAE 항 (작을수록 좋음)   └─ FICR 항 (클수록 좋으므로 빼 준다)

    입력:
        pred : (n,) 예측 이용률 [0,1]
        y    : (n,) 실제 이용률 [0,1]. **채점 대상 행만** 넣는다.
        ficr_weight (λ) : FICR 항의 가중치. **기본 1.0이 대회 산식 그대로다.**

    ## λ를 1보다 크게 두는 근거 (2026-08-03, v5~v7 리더보드 관찰)

    점수는 `0.5·(1-NMAE) + 0.5·FICR`로 두 항이 대등하지만, **남은 여지는 전혀 대등하지 않다.**
    현재 우리 1-NMAE는 **0.869**로 이미 1에 가깝고 FICR은 **0.414**로 멀리 있다.
    그리고 v5→v7에서 MLP 비중을 올리자 **1-NMAE는 오르다 포화(0.8671→0.8688)하고
    FICR은 단조 하락(0.4155→0.4123)** 했다 — 산식 손실로 학습했는데도 모델이
    **NMAE 쪽으로 치우쳐** 있다는 뜻이다.

    λ>1은 "NMAE를 조금 내주고 밴드 안으로 들어가라"는 지시다.
    ⚠️ λ≠1은 **대회 산식이 아닌 다른 목적함수**를 최적화하는 것이므로, 반드시 실제 점수
    (`group_score`, 즉 계단 그대로)로 검증해서 이득이 있을 때만 쓴다.

    ⚠️ 주의: `-metric_loss`는 score가 아니라 **score - 0.5** 다.
       score = 0.5·(1 - nmae) + 0.5·ficr = **0.5** - 0.5·nmae + 0.5·ficr 이기 때문.
       상수 0.5는 기울기에 영향이 없어 학습에는 무해하지만,
       점수와 비교할 때는 반드시 `soft_group_score()`를 쓸 것.
    """
    e = torch.abs(pred - y)
    nmae = e.mean()
    ficr = (y * soft_price(e, t_soft)).sum() / (4.0 * y.sum() + 1e-8)
    return 0.5 * nmae - 0.5 * ficr_weight * ficr


def soft_group_score(pred: torch.Tensor, y: torch.Tensor, t_soft: float = T_SOFT) -> float:
    """`metric_loss`가 실제로 최적화하고 있는 '점수'. 검증·디버깅용.

    soft_group_score = 0.5 - metric_loss  ≈ group_score(계단 그대로)

    계단의 모서리를 둥글게 깎았으므로 **항상 실제 점수보다 조금 낮게** 나온다
    (밴드 경계 근처 표본의 단가를 4/3이 아니라 3.6/2.7처럼 매기기 때문).
    T가 작을수록 차이가 줄어든다.
    """
    return float(0.5 - metric_loss(pred, y, t_soft))


def group_score(actual_ratio: np.ndarray, pred_ratio: np.ndarray) -> float:
    """한 그룹의 대회 점수(이용률 단위, 계단 그대로). 조기 종료 기준으로 쓴다.

    src/metric.py와 동일한 계산을 그룹 하나에 대해 수행한다.
    (전체 점수 = 그룹별 점수의 평균이라는 항등식은 15-3절에서 검산했다.)
    입력은 **채점 대상 행만** 걸러진 상태여야 한다.
    """
    a = np.asarray(actual_ratio, dtype=float)
    e = np.abs(np.asarray(pred_ratio, dtype=float) - a)
    price = np.select([e <= BAND_FULL, e <= BAND_PART], [4.0, 3.0], default=0.0)
    return 0.5 * (1.0 - e.mean()) + 0.5 * (a * price).sum() / (a * 4.0).sum()


def fit_standardizer(X: np.ndarray):
    """표준화 통계(평균/표준편차). **반드시 학습 데이터에서만 호출한다**(누수 방지)."""
    return X.mean(axis=0), X.std(axis=0) + 1e-6


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> torch.Tensor:
    """학습에서 구한 (mu, sd)로 표준화해 텐서로. 검증/test에는 transform만 적용된다."""
    return torch.tensor(((X - mu) / sd).astype(np.float32))


def train_metric_mlp(X_tr: np.ndarray, y_tr: np.ndarray, seed: int,
                     n_epochs: int = MAX_EPOCHS, t_soft: float = T_SOFT,
                     eval_fn=None, hidden=HIDDEN, p_drop: float = DROPOUT,
                     lr: float = LR, weight_decay: float = WEIGHT_DECAY,
                     patience: int = PATIENCE, eval_every: int = EVAL_EVERY,
                     ficr_weight: float = 1.0, deterministic: bool = True):
    """산식 손실로 MLP를 학습한다 (full-batch AdamW + 코사인 스케줄 + 기울기 클리핑).

    입력:
        X_tr    : (n, d) **표준화된** 학습 피처
        y_tr    : (n,) 학습 타깃 = 이용률 [0,1]. **채점 대상 행만.**
        eval_fn : (model) -> float. 주어지면 EVAL_EVERY 에폭마다 호출해
                  값이 가장 큰 시점의 가중치로 되돌린다(조기 종료).
                  ⚠️ 이 값은 **대회 산식**이어야 한다 — 산식을 최적화하니 산식으로 멈춰야 한다.
    출력: (model, best_epoch)
    """
    set_seed(seed, deterministic=deterministic)
    Xt = torch.tensor(X_tr.astype(np.float32))
    yt = torch.tensor(y_tr.astype(np.float32))

    model = MetricMLP(X_tr.shape[1], hidden, p_drop)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_score, best_state, best_epoch, bad = -np.inf, None, n_epochs, 0
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        metric_loss(model(Xt), yt, t_soft, ficr_weight).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        if eval_fn is not None and (ep % eval_every == 0 or ep == n_epochs - 1):
            model.eval()
            s = eval_fn(model)
            if s > best_score:
                best_score, bad, best_epoch = s, 0, ep + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += eval_every
                if bad >= patience:
                    break

    if eval_fn is not None and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


class WindMLP(tnn.Module):
    """풍속 추정용 MLP (18-4절). 발전량용 `MetricMLP`와는 목적도 손실도 다르다.

    이쪽은 **SCADA 실측 풍속**을 맞히는 회귀이고 손실은 평범한 MSE다.
    LightGBM 풍속 모델과 **오차 방향이 달라서**(잔차 상관 0.851) 블렌드 자원으로 쓴다.
    """

    def __init__(self, n_features, hidden=(512, 256, 128), p_drop=0.15):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [tnn.Linear(prev, h), tnn.BatchNorm1d(h), tnn.ReLU(), tnn.Dropout(p_drop)]
            prev = h
        layers += [tnn.Linear(prev, 1)]
        self.net = tnn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_wind_mlp(X_tr_df, y, tr_mask, es_mask, infer_frames, seed=42,
                 max_epochs=60, batch_size=512, patience=8, verbose=True):
    """풍속 MLP를 학습하고 주어진 프레임들에 대한 추정 풍속을 돌려준다.

    입력:
        X_tr_df      : 학습용 입력 DataFrame (전체 train 기간)
        y            : (n,) SCADA 실측 풍속 Series
        tr_mask/es_mask : 학습 / 조기종료 구간 불리언 마스크
        infer_frames : 추론할 DataFrame 목록 (예: [train_X, test_X])
    출력: infer_frames와 같은 길이의 numpy 배열 리스트

    ⚠️ 표준화 통계(mu, sd)와 타깃 정규화(ymu, ysd)는 **학습 구간에서만** 구한다.
    ⚠️ 여기서는 미니배치를 쓴다 — MSE 손실은 배치 평균이라 흔들리지 않는다.
       (`metric_loss`의 FICR 항만 full-batch가 필요하다.)
    """
    mu, sd = X_tr_df.loc[tr_mask].mean(), X_tr_df.loc[tr_mask].std().replace(0, 1)
    ymu, ysd = float(y[tr_mask].mean()), float(y[tr_mask].std())

    def tx(d):
        return torch.tensor(((d - mu) / sd).to_numpy(), dtype=torch.float32)

    Xtr, Xes = tx(X_tr_df.loc[tr_mask]), tx(X_tr_df.loc[es_mask])
    ytr = torch.tensor(((y[tr_mask] - ymu) / ysd).to_numpy(), dtype=torch.float32)
    yes = torch.tensor(((y[es_mask] - ymu) / ysd).to_numpy(), dtype=torch.float32)

    torch.manual_seed(seed)
    model = WindMLP(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_b = max(1, int(np.ceil(len(Xtr) / batch_size)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs * n_b)
    loss_fn = tnn.MSELoss()

    best, best_state, left = float("inf"), None, patience
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for b in range(n_b):
            sel = perm[b * batch_size:(b + 1) * batch_size]
            if len(sel) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(Xtr[sel]), ytr[sel]).backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xes), yes).item()
        if v < best - 1e-5:
            best, best_state, left = v, {k: t.clone() for k, t in model.state_dict().items()}, patience
        else:
            left -= 1
            if left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"    windmlp: {ep + 1}에폭 (검증MSE {best:.4f})")

    outs = []
    with torch.no_grad():
        for df_ in infer_frames:
            X_ = tx(df_)
            o = np.concatenate([model(X_[i:i + 4096]).numpy() for i in range(0, len(X_), 4096)])
            outs.append(np.clip(o * ysd + ymu, 0.0, None))
    return outs


@torch.no_grad()
def predict_ratio(model: MetricMLP, X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """표준화 -> 예측. 반환은 **이용률** [0,1] (kWh 환산은 호출부에서 capacity를 곱한다)."""
    model.eval()
    out = []
    Xs = standardize(X, mu, sd)
    for i in range(0, len(Xs), 8192):          # 메모리 여유를 위해 나눠서 추론
        out.append(model(Xs[i:i + 8192]).numpy())
    return np.clip(np.concatenate(out), 0.0, 1.0)
