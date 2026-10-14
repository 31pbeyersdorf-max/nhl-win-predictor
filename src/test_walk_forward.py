
"""Walk-forward evaluation with annual model retraining."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
)

from pipeline import (
    MODEL_COLS,
    load_games,
    build_model_table,
    prepare,
    fit_model,
)

TEST_START_SEASON = 2022
OUTPUT_PATH = Path("predictions/walk_forward_metrics.csv")


def main():
    games = load_games()

    table, _, _ = build_model_table(games)
    prepared = prepare(table).copy()

    prepared["season"] = prepared["season"].astype(int)

    # Exclude future-dated records.
    cutoff = pd.Timestamp.now().normalize()
    prepared = prepared[
        prepared["gameDate"] <= cutoff
    ].copy()

    seasons = sorted(
        prepared.loc[
            prepared["season"] >= TEST_START_SEASON,
            "season"
        ].unique()
    )

    metrics = []
    predictions = []

    for season in seasons:
        train = prepared[
            prepared["season"] < season
        ].copy()

        test = prepared[
            prepared["season"] == season
        ].copy()

        if train.empty or test.empty:
            continue

        model = fit_model(train)

        probabilities = model.predict_proba(
            test[MODEL_COLS]
        )[:, 1]

        predicted = (probabilities >= 0.5).astype(int)

        metrics.append({
            "season": season,
            "games": len(test),
            "accuracy": accuracy_score(
                test["home_win"], predicted
            ),
            "log_loss": log_loss(
                test["home_win"],
                probabilities,
                labels=[0, 1]
            ),
            "brier": brier_score_loss(
                test["home_win"],
                probabilities
            ),
        })

        predictions.append(
            pd.DataFrame({
                "gameId": test["gameId"].to_numpy(),
                "season": season,
                "gameDate": test["gameDate"].to_numpy(),
                "actual_home_win": test["home_win"].to_numpy(),
                "predicted_home_win": predicted,
                "home_win_prob": probabilities,
            })
        )

    if not metrics:
        raise ValueError("No eligible seasons found.")

    summary = pd.DataFrame(metrics)
    all_predictions = pd.concat(
        predictions, ignore_index=True
    )

    overall = {
        "season": "All test seasons",
        "games": len(all_predictions),
        "accuracy": accuracy_score(
            all_predictions["actual_home_win"],
            all_predictions["predicted_home_win"]
        ),
        "log_loss": log_loss(
            all_predictions["actual_home_win"],
            all_predictions["home_win_prob"],
            labels=[0, 1]
        ),
        "brier": brier_score_loss(
            all_predictions["actual_home_win"],
            all_predictions["home_win_prob"]
        ),
    }

    summary = pd.concat(
        [pd.DataFrame([overall]), summary],
        ignore_index=True
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    summary.to_csv(OUTPUT_PATH, index=False)

    all_predictions.to_csv(
        "predictions/walk_forward_predictions.csv",
        index=False
    )

    print("\nWalk-forward backtest:")
    print(summary.round(4).to_string(index=False))

    print("\nSaved walk-forward results.")


if __name__ == "__main__":
    main()
