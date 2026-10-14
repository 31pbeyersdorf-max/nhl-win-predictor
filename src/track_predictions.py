
"""Grade only timestamp-verifiable pregame predictions against NHL final scores.
Run from project root: python src/track_predictions.py
"""
from pathlib import Path
import pandas as pd
import requests

PREDICTIONS = Path("predictions/predictions.csv")
RESULTS = Path("predictions/verified_live_results.csv")
METRICS = Path("predictions/verified_live_metrics.csv")
API = "https://api-web.nhle.com/v1/gamecenter/{}/landing"


def fetch_result(game_id):
    try:
        response = requests.get(API.format(game_id), timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("gameState") not in {"OFF", "FINAL"}:
            return None
        home = data.get("homeTeam", {})
        away = data.get("awayTeam", {})
        hs, aws = home.get("score"), away.get("score")
        if hs is None or aws is None or hs == aws:
            return None
        return {"game_id": str(game_id), "actual_home_win": int(hs > aws),
                "actual_home_goals": int(hs), "actual_away_goals": int(aws)}
    except (requests.RequestException, ValueError) as error:
        print(f"Could not fetch {game_id}: {error}")
        return None


def main():
    if not PREDICTIONS.exists():
        print("No predictions/predictions.csv found.")
        return
    raw = pd.read_csv(PREDICTIONS, dtype={"game_id": str})
    if raw.empty:
        print("No predictions logged yet.")
        return
    for name in ("predicted_at_utc", "game_start_utc"):
        if name not in raw.columns:
            raw[name] = pd.NA
    raw["prediction_time"] = pd.to_datetime(
        raw["predicted_at_utc"], utc=True, errors="coerce")
    raw["start_time"] = pd.to_datetime(
        raw["game_start_utc"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    valid = raw.loc[
        raw["prediction_time"].notna()
        & raw["start_time"].notna()
        & (raw["prediction_time"] < raw["start_time"])
    ].sort_values("prediction_time").drop_duplicates("game_id", keep="first").copy()
    print(f"Verified pregame records: {len(valid)}; "
          f"legacy/unverified records: {len(raw) - len(valid)}")
    if valid.empty:
        print("No timestamp-verified pregame predictions yet. Run predict_upcoming.py before puck drop.")
        return

    eligible = valid.loc[valid["start_time"] <= now].copy()
    if RESULTS.exists():
        results = pd.read_csv(RESULTS, dtype={"game_id": str})
    else:
        results = pd.DataFrame(columns=["game_id", "actual_home_win",
                                        "actual_home_goals", "actual_away_goals"])
    existing = set(results["game_id"].astype(str))
    fetched = []
    for game_id in eligible["game_id"].astype(str):
        if game_id in existing:
            continue
        result = fetch_result(game_id)
        if result:
            fetched.append(result)
            existing.add(game_id)
    if fetched:
        results = pd.concat([results, pd.DataFrame(fetched)], ignore_index=True)
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(RESULTS, index=False)
    tracked = valid.merge(results, on="game_id", how="inner")
    if tracked.empty:
        print("No verified pregame predictions have finished yet.")
        return
    tracked["home_win_prob"] = pd.to_numeric(tracked["home_win_prob"], errors="coerce")
    tracked = tracked.dropna(subset=["home_win_prob"])
    tracked["actual_home_win"] = tracked["actual_home_win"].astype(int)
    tracked["predicted_home_win"] = (tracked["home_win_prob"] >= 0.5).astype(int)
    tracked["correct"] = (tracked["predicted_home_win"] == tracked["actual_home_win"]).astype(int)
    tracked["brier_error"] = (tracked["home_win_prob"] - tracked["actual_home_win"]) ** 2
    metrics = pd.DataFrame([{"completed_games": len(tracked),
                             "accuracy": tracked["correct"].mean(),
                             "brier": tracked["brier_error"].mean()}])
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS, index=False)
    print("\nVERIFIED LIVE PREDICTION TRACKER")
    print(f"Completed games: {len(tracked)}")
    print(f"Winner accuracy: {metrics.iloc[0]['accuracy']:.1%}")
    print(f"Brier score: {metrics.iloc[0]['brier']:.4f}")
    print(tracked.sort_values("start_time").tail(10)[
        ["game_date", "away_team", "home_team", "home_win_prob", "actual_home_win", "correct"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
