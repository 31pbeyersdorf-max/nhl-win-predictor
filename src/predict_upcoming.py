"""Predict upcoming NHL games and append the picks to predictions/predictions.csv.

Run from the project root:   python src/predict_upcoming.py
Re-download the MoneyPuck game-by-game file first so the data is current.
"""
import os
import sys
import time
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from pipeline import (STATS, WINDOW, MODEL_COLS, ELO_START, load_games,
                      build_model_table, prepare, fit_model,
                      elo_expected, regress_to_mean)

HORIZON_DAYS = 3                      # how many days ahead to predict
LOG_PATH = "predictions/predictions.csv"
API_URL = "https://api-web.nhle.com/v1/schedule/"

# The NHL API and MoneyPuck spell a few teams differently.
API_TO_DATA = {"TBL": "TB", "NJD": "NJ", "SJS": "SJ", "LAK": "LA"}
AVG_COLS = [s + "_avg10" for s in STATS]
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

    # 2. Each team's current form = average of its last 10 games
    form, last_played = {}, {}
    for team, grp in games.sort_values("gameDate").groupby("team"):
        recent = grp.tail(WINDOW)
        form[team] = {c: recent[s].mean() for c, s in zip(AVG_COLS, STATS)}
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
        for c in AVG_COLS:
            row["home_" + c] = form[h][c]
            row["away_" + c] = form[a][c]
        row["home_rest_days"] = rest_days(h, g["date"])
        row["away_rest_days"] = rest_days(a, g["date"])
        elo_p = elo_expected(cur.get(h, ELO_START), cur.get(a, ELO_START))
        row["elo_logit"] = np.log(elo_p / (1 - elo_p))
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

    print("\nPredictions (chance the HOME team wins in regulation/overtime):")
    for (g, elo_p), p in zip(meta, probs):
        print(f"  {g['date']}  {g['away']:>4} @ {g['home']:<4}  "
              f"{p:5.1%}   (Elo alone: {elo_p:5.1%})")

    # 6. Append to the log. Existing rows are never changed.
    new = pd.DataFrame([{
        "predicted_on": today.isoformat(),
        "game_id": g["game_id"],
        "game_date": g["date"].isoformat(),
        "away_team": g["away"],
        "home_team": g["home"],
        "home_win_prob": round(float(p), 4),
        "elo_prob": round(float(e), 4),
    } for (g, e), p in zip(meta, probs)])

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH)
        seen = set(zip(old["game_id"], old["predicted_on"]))
        keep = [(gid, po) not in seen
                for gid, po in zip(new["game_id"], new["predicted_on"])]
        new = new[keep]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(LOG_PATH, index=False)
    print(f"\nLogged {len(new)} new predictions to {LOG_PATH} "
          f"({len(out)} total).")


if __name__ == "__main__":
    main()