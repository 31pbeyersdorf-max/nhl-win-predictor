"""Compare logged predictions with real results.

Run after re-downloading the MoneyPuck file:   python src/grade_predictions.py
"""
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from pipeline import load_games

LOG_PATH = "predictions/predictions.csv"
GRADED_PATH = "predictions/graded.csv"


def main():
    preds = pd.read_csv(LOG_PATH, parse_dates=["predicted_on", "game_date"])

    # For each game, use the LAST prediction made on or before game day
    preds = preds[preds["predicted_on"] <= preds["game_date"]]
    preds = preds.sort_values("predicted_on").groupby("game_id").tail(1)

    games = load_games()
    results = (games[games["home_or_away"] == "HOME"][["gameId", "goal_diff"]]
               .rename(columns={"gameId": "game_id"}))
    merged = preds.merge(results, on="game_id", how="left")

    pending = merged["goal_diff"].isna().sum()
    shootouts = (merged["goal_diff"] == 0).sum()
    graded = merged[merged["goal_diff"].notna() & (merged["goal_diff"] != 0)].copy()
    graded["home_win"] = (graded["goal_diff"] > 0).astype(int)

    print(f"Games predicted:        {len(merged)}")
    print(f"  still pending:        {pending}  (or not in your data file yet)")
    print(f"  shootouts (ungraded): {shootouts}")
    print(f"  graded:               {len(graded)}")
    if graded.empty:
        print("\nNothing to grade yet. Re-download the MoneyPuck file after games finish.")
        return

    y = graded["home_win"]
    p = graded["home_win_prob"]
    e = graded["elo_prob"]
    print(f"\nModel         | accuracy {accuracy_score(y, p > 0.5):.3f} "
          f"| log loss {log_loss(y, p, labels=[0, 1]):.4f}")
    print(f"Elo alone     | accuracy {accuracy_score(y, e > 0.5):.3f} "
          f"| log loss {log_loss(y, e, labels=[0, 1]):.4f}")
    print(f"Always home   | accuracy {y.mean():.3f}")
    if len(graded) < 100:
        print(f"\nOnly {len(graded)} graded games so far. Accuracy on a sample this "
              "small is mostly noise; check back as it grows.")

    graded.to_csv(GRADED_PATH, index=False)
    print(f"\nSaved {GRADED_PATH}")


if __name__ == "__main__":
    main()