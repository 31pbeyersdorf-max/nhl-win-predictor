
"""Tune Elo settings using the 2021 season for validation."""

from itertools import product

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
)

import pipeline

TRAIN_END = 2020
VALIDATION_SEASON = 2021

K_VALUES = [4, 6, 10, 15, 20]
HOME_ADV_VALUES = [0, 25, 50, 75, 100]


def test_elo(games, k, home_adv):
    original_k = pipeline.ELO_K
    original_home_adv = pipeline.ELO_HOME_ADV

    try:
        pipeline.ELO_K = k
        pipeline.ELO_HOME_ADV = home_adv

        table, _, _ = pipeline.build_model_table(games)
        prepared = pipeline.prepare(table)

        train = prepared[
            prepared["season"] <= TRAIN_END
        ].copy()

        validation = prepared[
            prepared["season"] == VALIDATION_SEASON
        ].copy()

        if train.empty or validation.empty:
            raise ValueError("Missing training or validation games.")

        model = pipeline.fit_model(train)

        probabilities = model.predict_proba(
            validation[pipeline.MODEL_COLS]
        )[:, 1]

        predicted = (probabilities >= 0.5).astype(int)

        return {
            "elo_k": k,
            "home_adv": home_adv,
            "games": len(validation),
            "accuracy": accuracy_score(
                validation["home_win"], predicted
            ),
            "log_loss": log_loss(
                validation["home_win"],
                probabilities,
                labels=[0, 1]
            ),
            "brier": brier_score_loss(
                validation["home_win"],
                probabilities
            ),
        }

    finally:
        pipeline.ELO_K = original_k
        pipeline.ELO_HOME_ADV = original_home_adv


def main():
    games = pipeline.load_games()

    results = []

    for k, home_adv in product(K_VALUES, HOME_ADV_VALUES):
        print(f"Testing Elo K={k}, Home Advantage={home_adv}")

        result = test_elo(games, k, home_adv)
        results.append(result)

    summary = pd.DataFrame(results)

    summary = summary.sort_values(
        ["log_loss", "accuracy"],
        ascending=[True, False]
    )

    print("\nTop 10 Elo configurations:")
    print(summary.head(10).round(4).to_string(index=False))

    best = summary.iloc[0]

    print(
        f"\nBest validation configuration: "
        f"K={int(best['elo_k'])}, "
        f"Home Advantage={int(best['home_adv'])}"
    )

    summary.to_csv(
        "predictions/elo_tuning_results.csv",
        index=False
    )


if __name__ == "__main__":
    main()
