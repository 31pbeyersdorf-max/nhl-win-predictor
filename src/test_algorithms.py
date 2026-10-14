
"""Compare NHL winner prediction algorithms on validation data."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import (
    MODEL_COLS,
    build_model_table,
    load_games,
    prepare,
)

TRAIN_END = 2020
VALIDATION_SEASON = 2021

OUTPUT = Path("predictions/algorithm_comparison.csv")


def main():
    games = load_games()
    table, _, _ = build_model_table(games)
    prepared = prepare(table).copy()

    train = prepared[
        prepared["season"] <= TRAIN_END
    ].copy()

    validation = prepared[
        prepared["season"] == VALIDATION_SEASON
    ].copy()

    if train.empty or validation.empty:
        raise ValueError("Missing training or validation data.")

    X_train = train[MODEL_COLS]
    y_train = train["home_win"]

    X_val = validation[MODEL_COLS]
    y_val = validation["home_win"]

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    results = []

    print(f"\nTraining games: {len(train)}")
    print(f"Validation games: {len(validation)}")

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results.append({
            "model": name,
            "games": len(validation),
            "accuracy": accuracy_score(y_val, predictions),
            "log_loss": log_loss(
                y_val, probabilities, labels=[0, 1]
            ),
            "brier": brier_score_loss(y_val, probabilities),
        })

    summary = pd.DataFrame(results).sort_values(
        ["log_loss", "accuracy"],
        ascending=[True, False],
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)

    print("\nALGORITHM COMPARISON")
    print("=" * 55)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved results to {OUTPUT}")


if __name__ == "__main__":
    main()
