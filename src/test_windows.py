
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

import pipeline

TRAIN_END = 2020
VALIDATION_SEASON = 2021

WINDOWS = [5, 10, 15, 20, 30]


def test_window(games, window):
    original_window = pipeline.WINDOW
    original_min_games = pipeline.MIN_GAMES

    try:
        pipeline.WINDOW = window
        pipeline.MIN_GAMES = max(3, window // 2)

        table, _, _ = pipeline.build_model_table(games)
        prepared = pipeline.prepare(table)

        train = prepared[
            prepared["season"] <= TRAIN_END
        ].copy()

        validation = prepared[
            prepared["season"] == VALIDATION_SEASON
        ].copy()

        if train.empty or validation.empty:
            raise ValueError("Training or validation data is empty.")

        model = pipeline.fit_model(train)

        probabilities = model.predict_proba(
            validation[pipeline.MODEL_COLS]
        )[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        return {
            "window": window,
            "games": len(validation),
            "accuracy": accuracy_score(
                validation["home_win"], predictions
            ),
            "log_loss": log_loss(
                validation["home_win"],
                probabilities,
                labels=[0, 1]
            ),
            "brier": brier_score_loss(
                validation["home_win"],
                probabilities
            )
        }

    finally:
        pipeline.WINDOW = original_window
        pipeline.MIN_GAMES = original_min_games


def main():
    games = pipeline.load_games()

    results = []

    for window in WINDOWS:
        print(f"Testing {window}-game rolling window...")

        result = test_window(games, window)
        results.append(result)

    summary = pd.DataFrame(results)
    summary = summary.sort_values(
        ["log_loss", "accuracy"],
        ascending=[True, False]
    )

    print("\nRolling Window Experiment:")
    print(summary.round(4).to_string(index=False))

    best = summary.iloc[0]

    print(
        f"\nBest validation window: {int(best['window'])} games"
    )


if __name__ == "__main__":
    main()
