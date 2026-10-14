
"""Chronological NHL probability recalibration experiment."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "predictions/backtest_predictions.csv"
OUTPUT = ROOT / "predictions/recalibration_comparison.csv"
DETAILS = ROOT / "predictions/recalibration_predictions.csv"


def evaluate(name, actual, probabilities):
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)

    return {
        "method": name,
        "games": len(actual),
        "brier_score": brier_score_loss(actual, probabilities),
        "log_loss": log_loss(actual, probabilities, labels=[0, 1]),
        "winner_accuracy": accuracy_score(
            actual, (probabilities >= 0.5).astype(int)
        ),
    }


def main():
    df = pd.read_csv(INPUT)

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce")
    df["home_win_prob"] = pd.to_numeric(
        df["home_win_prob"], errors="coerce"
    )

    df = df.dropna(
        subset=["season", "home_win", "home_win_prob"]
    ).copy()

    df = df[
        df["home_win"].isin([0, 1])
        & df["home_win_prob"].between(0, 1)
    ].copy()

    # Use earlier holdout seasons to learn the adjustment.
    calibration = df[df["season"].isin([2022, 2023])].copy()

    # Evaluate on later, untouched seasons.
    test = df[df["season"].isin([2024, 2025])].copy()

    if calibration.empty or test.empty:
        raise ValueError("Missing calibration or test seasons.")

    x_train = calibration["home_win_prob"].to_numpy()
    y_train = calibration["home_win"].to_numpy().astype(int)

    x_test = test["home_win_prob"].to_numpy()
    y_test = test["home_win"].to_numpy().astype(int)

    # Platt scaling: logistic regression on model probabilities.
    platt = LogisticRegression(max_iter=1000)
    platt.fit(x_train.reshape(-1, 1), y_train)
    platt_probs = platt.predict_proba(
        x_test.reshape(-1, 1)
    )[:, 1]

    # Isotonic regression: flexible monotonic adjustment.
    isotonic = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0,
        y_max=1,
    )
    isotonic.fit(x_train, y_train)
    isotonic_probs = isotonic.predict(x_test)

    comparison = pd.DataFrame([
        evaluate("Original model", y_test, x_test),
        evaluate("Platt scaling", y_test, platt_probs),
        evaluate("Isotonic regression", y_test, isotonic_probs),
    ])

    comparison.to_csv(OUTPUT, index=False)

    test["original_probability"] = x_test
    test["platt_probability"] = platt_probs
    test["isotonic_probability"] = isotonic_probs
    test.to_csv(DETAILS, index=False)

    print("\nCHRONOLOGICAL RECALIBRATION EXPERIMENT")
    print("=" * 70)
    print(f"Calibration games: {len(calibration):,}")
    print(f"Evaluation games: {len(test):,}")
    print("\n")
    print(comparison.round(4).to_string(index=False))

    print(f"\nSaved: {OUTPUT}")
    print(f"Saved: {DETAILS}")


if __name__ == "__main__":
    main()
