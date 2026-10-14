"""Goal-count extension for the NHL predictor.

Uses MoneyPuck's recorded final goals (including any shootout deciding goal).
Independent Poisson goal models are a baseline, not an NHL overtime simulator.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import MODEL_COLS


def make_goal_table(table, games):
    """Add actual home and away goals to the pregame feature table."""

    home = (
        games[games["home_or_away"] == "HOME"]
        [["gameId", "goalsFor"]]
        .drop_duplicates(subset=["gameId"])
        .rename(columns={"goalsFor": "home_goals"})
    )

    away = (
        games[games["home_or_away"] == "AWAY"]
        [["gameId", "goalsFor"]]
        .drop_duplicates(subset=["gameId"])
        .rename(columns={"goalsFor": "away_goals"})
    )

    goal_table = (
        table.merge(home, on="gameId", validate="one_to_one")
        .merge(away, on="gameId", validate="one_to_one")
    )

    return goal_table.dropna(
        subset=MODEL_COLS + ["home_goals", "away_goals"]
    )



def fit_score_models(goal_table):
    """Fit one Poisson model for each side using existing pregame features."""
    models = {}
    for side in ("home", "away"):
        model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000))
        model.fit(goal_table[MODEL_COLS], goal_table[f"{side}_goals"])
        models[side] = model
    return models



def predict_score(models, features, max_goals=15):
    """Estimate scoring probabilities using independent Poisson models."""

    if isinstance(features, dict):
        features = pd.DataFrame([features])

    x = features[MODEL_COLS]

    home_xg = float(models["home"].predict(x)[0])
    away_xg = float(models["away"].predict(x)[0])

    scores = np.arange(max_goals + 1)

    home_probs = poisson.pmf(scores, home_xg)
    away_probs = poisson.pmf(scores, away_xg)

    # Probability of each possible score combination
    joint = np.outer(home_probs, away_probs)
    joint = joint / joint.sum()

    # Most likely score, including ties in the baseline distribution
    home_goals, away_goals = np.unravel_index(
        joint.argmax(), joint.shape
    )

    # Baseline outcome probabilities
    home_win_prob = np.tril(joint, k=-1).sum()
    away_win_prob = np.triu(joint, k=1).sum()
    tie_prob = np.trace(joint)

    # Expected total goals
    expected_total = home_xg + away_xg

    # Probability of more than 5.5 goals
    over_5_5 = 1 - poisson.cdf(5, expected_total)

    return {
        "expected_home_goals": round(home_xg, 2),
        "expected_away_goals": round(away_xg, 2),
        "expected_total_goals": round(expected_total, 2),

        "most_likely_home_score": int(home_goals),
        "most_likely_away_score": int(away_goals),

        "home_win_reg_prob": round(float(home_win_prob), 4),
        "away_win_reg_prob": round(float(away_win_prob), 4),
        "tie_reg_prob": round(float(tie_prob), 4),

        "over_5_5_prob": round(float(over_5_5), 4),
    }
