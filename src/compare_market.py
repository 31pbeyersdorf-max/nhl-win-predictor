
"""Compare NHL predictions with timestamped sportsbook prices."""

from pathlib import Path

import pandas as pd

from market_analysis import (
    implied_probability,
    no_vig_probabilities,
    expected_value,
)

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "predictions/predictions.csv"
ODDS = ROOT / "data/market_odds.csv"
OUTPUT = ROOT / "predictions/market_comparison.csv"


def main():
    predictions = pd.read_csv(
        PREDICTIONS, dtype={"game_id": str}
    )

    odds = pd.read_csv(
        ODDS, dtype={"game_id": str}
    )

    # Normalize timestamps.
    for column in ["predicted_at_utc", "game_start_utc"]:
        predictions[column] = pd.to_datetime(
            predictions[column], utc=True, errors="coerce"
        )

    for column in ["recorded_at_utc", "bookmaker_last_update_utc"]:
        odds[column] = pd.to_datetime(
            odds[column], utc=True, errors="coerce"
        )

    # Keep only genuinely timestamped pregame predictions.
    predictions = predictions[
        predictions["predicted_at_utc"].notna()
        & predictions["game_start_utc"].notna()
        & (
            predictions["predicted_at_utc"]
            < predictions["game_start_utc"]
        )
    ].copy()

    predictions = predictions.drop_duplicates("game_id")

    # Join sportsbook observations to NHL game IDs.
    merged = predictions.merge(
        odds,
        on="game_id",
        how="inner",
        validate="one_to_many",
    )

    # Keep prices observed before puck drop.
    merged = merged[
        merged["recorded_at_utc"].notna()
        & (merged["recorded_at_utc"] < merged["game_start_utc"])
        & merged["home_odds"].notna()
        & merged["away_odds"].notna()
        & (merged["odds_type"] == "moneyline_full_game")
    ].copy()

    if merged.empty:
        print("No matching pregame odds and predictions.")
        return

    # Ensure numeric values.
    merged["home_win_prob"] = pd.to_numeric(
        merged["home_win_prob"], errors="coerce"
    )
    merged["home_odds"] = pd.to_numeric(
        merged["home_odds"], errors="coerce"
    )
    merged["away_odds"] = pd.to_numeric(
        merged["away_odds"], errors="coerce"
    )

    merged = merged.dropna(
        subset=["home_win_prob", "home_odds", "away_odds"]
    )

    merged["market_home_implied"] = merged["home_odds"].apply(
        implied_probability
    )

    merged["market_away_implied"] = merged["away_odds"].apply(
        implied_probability
    )

    markets = merged.apply(
        lambda row: no_vig_probabilities(
            row["home_odds"], row["away_odds"]
        ),
        axis=1,
    )

    merged["market_home_no_vig"] = markets.apply(
        lambda x: x["home"]
    )

    merged["market_overround"] = markets.apply(
        lambda x: x["overround"]
    )

    merged["home_model_edge"] = (
        merged["home_win_prob"] - merged["market_home_no_vig"]
    )

    merged["home_ev_per_10"] = merged.apply(
        lambda row: expected_value(
            row["home_win_prob"], row["home_odds"], 10
        ),
        axis=1,
    )

    merged["away_ev_per_10"] = merged.apply(
        lambda row: expected_value(
            1 - row["home_win_prob"], row["away_odds"], 10
        ),
        axis=1,
    )

    # One most recently observed price per sportsbook per game.
    merged = (
        merged.sort_values("recorded_at_utc")
        .drop_duplicates(
            ["game_id", "sportsbook"], keep="last"
        )
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)

    display_cols = [
        "away_team",
        "home_team",
        "sportsbook",
        "home_odds",
        "away_odds",
        "home_win_prob",
        "market_home_no_vig",
        "home_model_edge",
        "home_ev_per_10",
        "away_ev_per_10",
    ]

    print("\nMODEL VS. MARKET")
    print("=" * 80)
    print(
        merged[display_cols].round(3).to_string(index=False)
    )

    print(f"\nSaved {len(merged)} comparisons to {OUTPUT}")


if __name__ == "__main__":
    main()
