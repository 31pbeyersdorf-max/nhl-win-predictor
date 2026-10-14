
"""Track NHL model performance against timestamped sportsbook odds.

Run from project root:
    python src/track_market_performance.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent.parent

MARKET = ROOT / "predictions/market_comparison.csv"
RESULTS = ROOT / "predictions/verified_live_results.csv"

OUTPUT = ROOT / "predictions/market_performance.csv"
SUMMARY = ROOT / "predictions/market_performance_summary.csv"

# Use one consistent sportsbook for the primary benchmark.
PRIMARY_BOOK = "draftkings"
STAKE = 10.0


def decimal_odds(american):
    american = float(american)

    if american > 0:
        return 1 + american / 100

    if american < 0:
        return 1 + 100 / abs(american)

    raise ValueError("American odds cannot be zero.")


def main():
    if not MARKET.exists() or not RESULTS.exists():
        print(
            "Missing market comparisons or verified NHL results. "
            "Run compare_market.py and track_predictions.py first."
        )
        return

    market = pd.read_csv(MARKET, dtype={"game_id": str})
    results = pd.read_csv(RESULTS, dtype={"game_id": str})

    # Restrict to one sportsbook and one record per game.
    market = market[
        market["sportsbook"] == PRIMARY_BOOK
    ].copy()

    if market.empty:
        print(f"No observations for {PRIMARY_BOOK}.")
        return

    market["recorded_at_utc"] = pd.to_datetime(
        market["recorded_at_utc"],
        utc=True,
        errors="coerce",
    )

    market = (
        market.sort_values("recorded_at_utc")
        .drop_duplicates("game_id", keep="last")
    )

    results = results.drop_duplicates("game_id")

    tracked = market.merge(
        results[["game_id", "actual_home_win"]],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )

    if tracked.empty:
        print("No completed games with matching market odds yet.")
        return

    for col in [
        "home_win_prob",
        "market_home_no_vig",
        "home_ev_per_10",
        "away_ev_per_10",
        "home_odds",
        "away_odds",
    ]:
        tracked[col] = pd.to_numeric(
            tracked[col], errors="coerce"
        )

    tracked["actual_home_win"] = pd.to_numeric(
        tracked["actual_home_win"], errors="coerce"
    )

    tracked = tracked.dropna(
        subset=[
            "home_win_prob",
            "market_home_no_vig",
            "actual_home_win",
            "home_odds",
            "away_odds",
        ]
    )

    tracked = tracked[
        tracked["actual_home_win"].isin([0, 1])
        & tracked["home_win_prob"].between(0, 1)
        & tracked["market_home_no_vig"].between(0, 1)
    ].copy()

    if tracked.empty:
        print("No valid completed comparisons.")
        return

    # Winner prediction accuracy.
    tracked["model_pick_home"] = (
        tracked["home_win_prob"] >= 0.5
    ).astype(int)

    tracked["market_pick_home"] = (
        tracked["market_home_no_vig"] >= 0.5
    ).astype(int)

    tracked["model_correct"] = (
        tracked["model_pick_home"] == tracked["actual_home_win"]
    ).astype(int)

    tracked["market_correct"] = (
        tracked["market_pick_home"] == tracked["actual_home_win"]
    ).astype(int)

    # Paper-bet policy:
    # Take the side with positive theoretical EV.
    # If neither side has positive EV, make no bet.
    tracked["paper_side"] = np.select(
        [
            (tracked["home_ev_per_10"] > 0)
            & (
                tracked["home_ev_per_10"]
                >= tracked["away_ev_per_10"]
            ),
            tracked["away_ev_per_10"] > 0,
        ],
        ["HOME", "AWAY"],
        default="NO BET",
    )

    def paper_profit(row):
        side = row["paper_side"]

        if side == "NO BET":
            return 0.0

        if side == "HOME":
            won = row["actual_home_win"] == 1
            odds = row["home_odds"]
        else:
            won = row["actual_home_win"] == 0
            odds = row["away_odds"]

        if won:
            return STAKE * (decimal_odds(odds) - 1)

        return -STAKE

    tracked["paper_profit"] = tracked.apply(
        paper_profit, axis=1
    )

    tracked["paper_stake"] = np.where(
        tracked["paper_side"] == "NO BET",
        0.0,
        STAKE,
    )

    tracked["model_brier_error"] = (
        tracked["home_win_prob"]
        - tracked["actual_home_win"]
    ) ** 2

    tracked["market_brier_error"] = (
        tracked["market_home_no_vig"]
        - tracked["actual_home_win"]
    ) ** 2

    tracked.to_csv(OUTPUT, index=False)

    total_staked = tracked["paper_stake"].sum()
    total_profit = tracked["paper_profit"].sum()

    summary = pd.DataFrame([{
        "sportsbook": PRIMARY_BOOK,
        "completed_games": len(tracked),
        "model_accuracy": tracked["model_correct"].mean(),
        "market_accuracy": tracked["market_correct"].mean(),
        "model_brier": brier_score_loss(
            tracked["actual_home_win"],
            tracked["home_win_prob"],
        ),
        "market_brier": brier_score_loss(
            tracked["actual_home_win"],
            tracked["market_home_no_vig"],
        ),
        "model_log_loss": log_loss(
            tracked["actual_home_win"],
            tracked["home_win_prob"].clip(1e-6, 1 - 1e-6),
            labels=[0, 1],
        ),
        "market_log_loss": log_loss(
            tracked["actual_home_win"],
            tracked["market_home_no_vig"].clip(1e-6, 1 - 1e-6),
            labels=[0, 1],
        ),
        "paper_bets": int(
            (tracked["paper_side"] != "NO BET").sum()
        ),
        "total_staked": total_staked,
        "paper_profit": total_profit,
        "paper_roi": (
            total_profit / total_staked
            if total_staked > 0 else np.nan
        ),
    }])

    summary.to_csv(SUMMARY, index=False)

    print("\nNHL MODEL VS. MARKET PERFORMANCE")
    print("=" * 70)
    print(summary.round(4).to_string(index=False))

    print("\nINDIVIDUAL GAMES")
    print(
        tracked[
            [
                "away_team",
                "home_team",
                "home_win_prob",
                "market_home_no_vig",
                "actual_home_win",
                "paper_side",
                "paper_profit",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {OUTPUT}")
    print(f"Saved: {SUMMARY}")


if __name__ == "__main__":
    main()
