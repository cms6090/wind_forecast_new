from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.metric import TARGET_COLS, CAPACITY_KWH

SAMPLE_SUBMISSION_PATH = Path("data/sample_submission.csv")
SUBMISSIONS_DIR = Path("submissions")
FILENAME_PATTERN = re.compile(r"^\d{8}_v\d+_.+\.csv$")


def build_submission(pred_df: pd.DataFrame, sample_path: Path = SAMPLE_SUBMISSION_PATH) -> pd.DataFrame:
    """pred_df(예측 대상 시각 + kpx_group_1/2/3 예측값)를 제출 양식에 맞춰 정렬·클리핑한다."""
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")

    merged = sample[["forecast_id", "forecast_kst_dtm"]].merge(
        pred_df[["forecast_kst_dtm", *TARGET_COLS]],
        on="forecast_kst_dtm",
        how="left",
    )

    for col in TARGET_COLS:
        merged[col] = merged[col].clip(lower=0, upper=CAPACITY_KWH[col])

    return merged[["forecast_id", "forecast_kst_dtm", *TARGET_COLS]]


def validate_submission(submission_df: pd.DataFrame, sample_path: Path = SAMPLE_SUBMISSION_PATH) -> None:
    """제출 직전 필수 검증. 하나라도 위반하면 AssertionError로 즉시 중단한다."""
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")

    assert list(submission_df.columns) == ["forecast_id", "forecast_kst_dtm", *TARGET_COLS], (
        f"컬럼 구성이 다릅니다: {list(submission_df.columns)}"
    )
    assert len(submission_df) == len(sample), (
        f"행 수가 다릅니다: {len(submission_df)} != {len(sample)}"
    )
    assert (submission_df["forecast_id"].to_numpy() == sample["forecast_id"].to_numpy()).all(), (
        "forecast_id가 sample_submission과 순서/값이 다릅니다."
    )
    assert (submission_df["forecast_kst_dtm"].to_numpy() == sample["forecast_kst_dtm"].to_numpy()).all(), (
        "forecast_kst_dtm이 sample_submission과 순서/값이 다릅니다."
    )
    assert not submission_df[TARGET_COLS].isna().any().any(), "예측값에 결측치가 있습니다."

    for col in TARGET_COLS:
        values = submission_df[col].to_numpy(dtype=float)
        assert (values >= 0).all(), f"{col}에 음수 예측값이 있습니다."
        assert (values <= CAPACITY_KWH[col]).all(), f"{col} 예측값이 설비용량({CAPACITY_KWH[col]} kWh)을 초과합니다."


def save_submission(submission_df: pd.DataFrame, filename: str, sample_path: Path = SAMPLE_SUBMISSION_PATH) -> Path:
    """검증 통과 후에만 submissions/ 아래 지정 파일명(YYYYMMDD_vN_설명.csv)으로 저장한다."""
    assert FILENAME_PATTERN.match(filename), (
        f"파일명 규칙(YYYYMMDD_vN_설명.csv) 위반: {filename}"
    )
    validate_submission(submission_df, sample_path=sample_path)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / filename
    submission_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
