"""Predict upcoming NHL games and append the picks to predictions/predictions.csv.
Run from the project root:   python src/predict_upcoming.py
Re-download the MoneyPuck game-by-game file first so the data is current.
"""
import os
import sys
import time
from collections import defaultdict
from datetime import date, timedelta, datetime, timezone
import numpy as np
import pandas as pd
import requests
from pipeline import (
    STATS, WINDOW, MIN_GAMES, MODEL_COLS, ELO_START,
    load_games, build_model_table, prepare, fit_model,
    elo_expected, regress_to_mean,
    add_rolling_features, add_opponent_adjusted_features,
)
from score_model import make_goal_table, fit_score_models, predict_score
HORIZON_DAYS = 3                      # how many days ahead to predict
LOG_PATH = "predictions/predictions.csv"
API_URL = "https://api-web.nhle.com/v1/schedule/"
# The NHL API and MoneyPuck spell a few teams differently.
API_TO_DATA = {"TBL": "TB", "NJD": "NJ", "SJS": "SJ", "LAK": "LA"}
AVG_COLS = [s + "_avg10" for s in STATS]
ADJUSTED_COLS = [
    "adjusted_xgf_avg15",
    "adjusted_xga_avg15",
]
FINISHED = {"OFF", "FINAL"}
NOT_STARTED = {"FUT", "PRE"}
def get_today():
    return date.today()
def get_json(url):
    resp = requests.get(
        url,
        headers={"User-Agent": "nhl-win-predictor (personal portfolio project)"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
def fetch_schedule(start, end):
    """Regular-season games from start to end (dates), from the NHL API."""
    found, cursor = {}, start
    while cursor <= end:
        data = get_json(API_URL + cursor.isoformat())
        days = data.get("gameWeek") or []
        if not days:
            break
        try:
            for day in days:
                d = date.fromisoformat(day["date"])
                for g in day.get("games", []):
                    if g.get("gameType") != 2:       # 2 = regular season
                        continue
                    found[g["id"]] = {
                        "game_id": g["id"],
                        "date": d,
                        "start_time_utc": g.get("startTimeUTC"),
                        "state": g.get("gameState"),
                        "sched": g.get("gameScheduleState", "OK"),
                        "home": g["homeTeam"]["abbrev"],
                        "away": g["awayTeam"]["abbrev"],
                    }
            next_cursor = date.fromisoformat(days[-1]["date"]) + timedelta(days=1)
        except (KeyError, TypeError, ValueError) as err:
            sys.exit(
                f"The NHL API response wasn't in the format I expected: {err!r}\n"
                "Paste this message to Claude so the script can be adjusted."
            )
        cursor = max(next_cursor, cursor + timedelta(days=1))
        time.sleep(0.5)                              # be polite to the API
    return [g for g in found.values() if start <= g["date"] <= end]
def season_of(d):
    """The season label MoneyPuck uses: the year the season started."""
    return d.year if d.month >= 9 else d.year - 1
def to_data_code(api_code, data_teams):
    for candidate in (api_code, API_TO_DATA.get(api_code)):
        if candidate in data_teams:
            return candidate
    return None
def main():
    games = load_games()
    data_teams = set(games["team"])
    data_end = games["gameDate"].max().date()
    today = get_today()
    print(f"MoneyPuck data runs through {data_end}. Today is {today}.")
    if (today - data_end).days > 2:
        print("Note: your data file looks out of date. Re-download it from "
              "MoneyPuck for the best predictions.")
    # 1. Train on every finished game in the data file
    table, ratings, last_season = build_model_table(games)
    model = fit_model(prepare(table))
    goal_table = make_goal_table(table, games)
    score_models = fit_score_models(goal_table)
    # 2. Each team's current form = average of its last 10 games
    form, last_played = {}, {}
    # Calculate historical opponent-adjusted performance.
    historical = add_opponent_adjusted_features(
        add_rolling_features(games)
    )
    for team, grp in historical.groupby("team"):
        grp = grp.sort_values(["gameDate", "gameId"])
        recent = grp.tail(WINDOW)
        team_form = {
            c: recent[s].mean()
            for c, s in zip(AVG_COLS, STATS)
        }
        # Recreate the pregame rolling opponent-adjusted
        # features from the previous WINDOW completed games.
        for raw_col, feature_col in [
            ("adjusted_xgf", "adjusted_xgf_avg15"),
            ("adjusted_xga", "adjusted_xga_avg15"),
        ]:
            values = recent[raw_col]
            team_form[feature_col] = (
                values.mean()
                if values.notna().sum() >= MIN_GAMES
                else np.nan
            )
        form[team] = team_form
        last_played[team] = grp["gameDate"].iloc[-1].date()
    # 3. Ask the NHL what is scheduled
    sched = fetch_schedule(data_end + timedelta(days=1),
                           today + timedelta(days=HORIZON_DAYS))
    sched = [g for g in sched if g["sched"] == "OK"]
    for g in sched:
        g["home_d"] = to_data_code(g["home"], data_teams)
        g["away_d"] = to_data_code(g["away"], data_teams)
    # Teams that played a finished game missing from the data file are "stale"
    stale = set()
    for g in sched:
        if g["state"] in FINISHED:
            stale.update([g["home_d"], g["away_d"]])
    # Dates of every unfinished game, used to work out rest days
    timeline = defaultdict(list)
    for g in sched:
        if g["state"] not in FINISHED:
            timeline[g["home_d"]].append(g["date"])
            timeline[g["away_d"]].append(g["date"])
    def rest_days(team, d):
        earlier = [x for x in timeline[team] if x < d] + [last_played[team]]
        return min((d - max(earlier)).days, 7)
    upcoming = sorted(
        (g for g in sched if g["state"] in NOT_STARTED and g["date"] >= today),
        key=lambda g: (g["date"], g["game_id"]),
    )
    if not upcoming:
        print(f"No upcoming games found in the next {HORIZON_DAYS} days.")
        return
    cur = dict(ratings)
    if season_of(upcoming[0]["date"]) != last_season:
        regress_to_mean(cur)                         # a new season has started
    # 4. Build one row of features per game
    rows, meta, skipped = [], [], []
    for g in upcoming:
        h, a = g["home_d"], g["away_d"]
        if h is None or a is None:
            skipped.append((g, "team code not found in your data"))
            continue
        if h in stale or a in stale:
            skipped.append((g, "data file is missing a recently finished game"))
            continue
        row = {}
        for c in AVG_COLS + ADJUSTED_COLS:
            row["home_" + c] = form[h][c]
            row["away_" + c] = form[a][c]
        row["home_rest_days"] = rest_days(h, g["date"])
        row["away_rest_days"] = rest_days(a, g["date"])
        elo_p = elo_expected(cur.get(h, ELO_START), cur.get(a, ELO_START))
        row["elo_logit"] = np.log(elo_p / (1 - elo_p))
        if any(pd.isna(row[col]) for col in MODEL_COLS):
            skipped.append((g, "insufficient historical features"))
            continue
        rows.append(row)
        meta.append((g, elo_p))
    if skipped:
        print("\nSkipped:")
        for g, why in skipped:
            print(f"  {g['date']} {g['away']} @ {g['home']}: {why}")
    if not rows:
        print("\nNothing to predict. Try re-downloading the MoneyPuck file.")
        return
    # 5. Predict
    X = pd.DataFrame(rows)[MODEL_COLS]
    probs = model.predict_proba(X)[:, 1]
        # Predict expected goals and suggested scores
    score_predictions = [
        predict_score(score_models, X.iloc[[i]])
        for i in range(len(X))
    ]
    print("\nProjected NHL Scores:")
    for (game, _), score in zip(meta, score_predictions):
        print(
            f"\n{game['away']} @ {game['home']}"
            f"\n  Most likely baseline score: "
            f"{score['most_likely_away_score']}-"
            f"{score['most_likely_home_score']}"
            f"\n  Expected goals: "
            f"{score['expected_away_goals']:.2f}-"
            f"{score['expected_home_goals']:.2f}"
            f"\n  Expected total: {score['expected_total_goals']:.2f}"
            f"\n  Over 5.5 probability: {score['over_5_5_prob']:.1%}"
            f"\n  Baseline tie probability: {score['tie_reg_prob']:.1%}"
        )
    print("\nPredictions (chance the HOME team wins in regulation/overtime):")
    for (g, elo_p), p in zip(meta, probs):
        print(f"  {g['date']}  {g['away']:>4} @ {g['home']:<4}  "
              f"{p:5.1%}   (Elo alone: {elo_p:5.1%})")
    # 6. Save win probabilities and score predictions
    recorded_at = datetime.now(timezone.utc)
    new = pd.DataFrame([{
        "predicted_on": today.isoformat(),
        "predicted_at_utc": recorded_at.isoformat(),
        "game_start_utc": g["start_time_utc"],
        "game_id": g["game_id"],
        "game_date": g["date"].isoformat(),
        "away_team": g["away"],
        "home_team": g["home"],
        # Winner predictions
        "home_win_prob": round(float(p), 4),
        "elo_prob": round(float(e), 4),
        # Expected goals
        "expected_home_goals": score["expected_home_goals"],
        "expected_away_goals": score["expected_away_goals"],
        "expected_total_goals": score["expected_total_goals"],
        # Most likely baseline score
        "predicted_home_score": score["most_likely_home_score"],
        "predicted_away_score": score["most_likely_away_score"],
        # Scoring probabilities
        "over_5_5_prob": score["over_5_5_prob"],
        "baseline_tie_prob": score["tie_reg_prob"],
    } for (g, e), p, score in zip(meta, probs, score_predictions)])
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH, dtype={"game_id": str})
    else:
        old = pd.DataFrame()

    # Never overwrite an existing prediction. Legacy rows without a timestamp
    # remain in the CSV but cannot be counted as verified live predictions.
    if not old.empty and "predicted_at_utc" in old.columns:
        verified_ids = set(
            old.loc[
                pd.to_datetime(old["predicted_at_utc"], utc=True, errors="coerce").notna(),
                "game_id",
            ].astype(str)
        )
    else:
        verified_ids = set()

    # Only record predictions made strictly before the scheduled start.
    start_times = pd.to_datetime(new["game_start_utc"], utc=True, errors="coerce")
    before_start = start_times.notna() & (recorded_at < start_times)
    not_already_logged = ~new["game_id"].astype(str).isin(verified_ids)
    to_save = new.loc[before_start & not_already_logged].copy()

    if not to_save.empty:
        out = pd.concat([old, to_save], ignore_index=True)
        out.to_csv(LOG_PATH, index=False)
    else:
        out = old

    print(f"\nSaved {len(to_save)} NEW pregame predictions to {LOG_PATH}.")
    print(f"Already logged: {(~not_already_logged).sum()}; "
          f"missing/past start time: {(~before_start).sum()}.")
    print(f"Historical records preserved: {len(old)}; total records: {len(out)}.")

if __name__ == "__main__":
    main()
