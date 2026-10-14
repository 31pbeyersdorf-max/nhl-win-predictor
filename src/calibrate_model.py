
"""Evaluate NHL winner probability calibration.

Run from project root:
    python src/calibrate_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "predictions/backtest_predictions.csv"
OUTPUT = ROOT / "predictions/calibration_results.csv"
SUMMARY = ROOT / "predictions/calibration_summary.csv"

def main():
    df = pd.read_csv(INPUT)

    # Use only games with verified winner outcomes.
    df = df.dropna(
        subset=["home_win_prob", "home_win"]
    ).copy()

    df["home_win_prob"] = pd.to_numeric(
        df["home_win_prob"], errors="coerce"
    )
    df["home_win"] = pd.to_numeric(
        df["home_win"], errors="coerce"
    )

    df = df.dropna(
        subset=["home_win_prob", "home_win"]
    )

    df = df[
        df["home_win_prob"].between(0, 1)
        & df["home_win"].isin([0, 1])
    ].copy()

    if df.empty:
        raise ValueError("No valid winner predictions found.")

    # Probability buckets: 0-10%, 10-20%, etc.
    bins = np.linspace(0, 1, 11)

    df["probability_bucket"] = pd.cut(
        df["home_win_prob"],
        bins=bins,
        include_lowest=True,
    )

    calibration = (
        df.groupby("probability_bucket", observed=True)
        .agg(
            games=("home_win", "size"),
            average_predicted_probability=(
                "home_win_prob", "mean"
            ),
            actual_home_win_rate=("home_win", "mean"),
        )
        .reset_index()
    )

    calibration["calibration_gap"] = (
        calibration["actual_home_win_rate"]
        - calibration["average_predicted_probability"]
    )

    calibration["bucket"] = (
        calibration["probability_bucket"].astype(str)
    )

    calibration.to_csv(OUTPUT, index=False)

    # Overall probability-quality metrics.
    brier = brier_score_loss(
        df["home_win"], df["home_win_prob"]
    )
    loss = log_loss(
        df["home_win"], df["home_win_prob"],
        labels=[0, 1]
    )

    expected_calibration_error = (
        (
            calibration["games"] / len(df)
        )
        * calibration["calibration_gap"].abs()
    ).sum()

    summary = pd.DataFrame([{
        "evaluated_games": len(df),
        "brier_score": brier,
        "log_loss": loss,
        "expected_calibration_error": (
            expected_calibration_error
        ),
    }])

    summary.to_csv(SUMMARY, index=False)

    print("\nNHL MODEL CALIBRATION")
    print("=" * 65)
    print(f"Evaluated games: {len(df):,}")
    print(f"Brier score: {brier:.4f}")
    print(f"Log loss: {loss:.4f}")
    print(
        "Expected calibration error: "
        f"{expected_calibration_error:.2%}"
    )

    print("\nCALIBRATION BY PROBABILITY BUCKET")
    print("=" * 65)

    display = calibration[
        [
            "bucket",
            "games",
            "average_predicted_probability",
            "actual_home_win_rate",
            "calibration_gap",
        ]
    ].copy()

    print(display.round(3).to_string(index=False))

    print(f"\nSaved: {OUTPUT}")
    print(f"Saved: {SUMMARY}")


if __name__ == "__main__":
    main()
