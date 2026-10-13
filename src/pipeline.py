"""Shared code: load MoneyPuck data, build features, run Elo, train the model."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# >>> Change this to your real filename <<<
DATA_PATH = "data/all_teams.csv"

STATS = [
    "xGoalsFor", "xGoalsAgainst", "goalsFor", "goalsAgainst",
    "shotsOnGoalFor", "shotsOnGoalAgainst",
    "highDangerShotsFor", "highDangerShotsAgainst",
]
WINDOW = 10      # games in each rolling average
MIN_GAMES = 5    # need at least this many earlier games to make features

FEATURES = [s + "_avg10" for s in STATS] + ["rest_days"]
MODEL_COLS = (["home_" + f for f in FEATURES]
              + ["away_" + f for f in FEATURES]
              + ["elo_logit"])

ELO_K = 6
ELO_HOME_ADV = 50
ELO_START = 1500


def load_games(path=DATA_PATH):
    """One row per team per game: regular season, all situations."""
    cols = ["team", "season", "gameId", "home_or_away", "gameDate",
            "situation", "playoffGame"] + STATS
    df = pd.read_csv(path, usecols=cols)
    g = df[(df["situation"] == "all") & (df["playoffGame"] == 0)].copy()
    # MoneyPuck writes a few teams with dots (T.B, N.J). Remove them.
    g["team"] = g["team"].str.replace(".", "", regex=False)
    g["gameDate"] = pd.to_datetime(g["gameDate"].astype(str), format="%Y%m%d")
    g["goal_diff"] = g["goalsFor"] - g["goalsAgainst"]
    return g.sort_values(["team", "gameDate"]).reset_index(drop=True)


def add_rolling_features(games):
    """Average of each team's PREVIOUS 10 games (shift(1) prevents leakage)."""
    games = games.sort_values(["team", "gameDate"]).copy()
    for col in STATS:
        games[col + "_avg10"] = (
            games.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())
        )
    games["rest_days"] = (
        games.groupby("team")["gameDate"].diff().dt.days.clip(upper=7)
    )
    return games


def elo_expected(r_home, r_away):
    return 1 / (1 + 10 ** (-((r_home + ELO_HOME_ADV) - r_away) / 400))


def regress_to_mean(ratings):
    """New season: pull every team 25% of the way back to average."""
    for t in ratings:
        ratings[t] = 0.75 * ratings[t] + 0.25 * ELO_START


def run_elo(games):
    """Walk through every game in date order. Returns pre-game Elo
    probabilities, the final ratings, and the last season seen."""
    home = (games[games["home_or_away"] == "HOME"]
            [["gameId", "gameDate", "season", "team", "goal_diff"]]
            .rename(columns={"team": "home_team"}))
    away = (games[games["home_or_away"] == "AWAY"][["gameId", "team"]]
            .rename(columns={"team": "away_team"}))
    g = (home.merge(away, on="gameId")
         .sort_values(["gameDate", "gameId"]).reset_index(drop=True))

    ratings, probs, last_season = {}, [], None
    for row in g.itertuples():
        if last_season is not None and row.season != last_season:
            regress_to_mean(ratings)
        last_season = row.season

        rh = ratings.get(row.home_team, ELO_START)
        ra = ratings.get(row.away_team, ELO_START)
        p = elo_expected(rh, ra)
        probs.append(p)                      # saved BEFORE updating

        if row.goal_diff > 0:
            result = 1.0
        elif row.goal_diff < 0:
            result = 0.0
        else:
            result = 0.5                     # shootout: winner unknown
        ratings[row.home_team] = rh + ELO_K * (result - p)
        ratings[row.away_team] = ra - ELO_K * (result - p)

    g["elo_prob"] = probs
    return g[["gameId", "elo_prob"]], ratings, last_season


def build_model_table(games):
    """One row per game: home features, away features, Elo."""
    feat = add_rolling_features(games)
    home = (feat[feat["home_or_away"] == "HOME"]
            [["gameId", "season", "gameDate", "team", "goal_diff"] + FEATURES])
    away = feat[feat["home_or_away"] == "AWAY"][["gameId", "team"] + FEATURES]
    home = home.rename(columns={**{c: "home_" + c for c in FEATURES},
                                "team": "home_team"})
    away = away.rename(columns={**{c: "away_" + c for c in FEATURES},
                                "team": "away_team"})
    table = home.merge(away, on="gameId")

    elo_probs, ratings, last_season = run_elo(games)
    table = table.merge(elo_probs, on="gameId")
    table["elo_logit"] = np.log(table["elo_prob"] / (1 - table["elo_prob"]))
    return table, ratings, last_season


def prepare(table):
    """Drop shootouts and rows without enough history; add the target."""
    t = table[table["goal_diff"] != 0].dropna(subset=MODEL_COLS).copy()
    t["home_win"] = (t["goal_diff"] > 0).astype(int)
    return t


def fit_model(prepared):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(prepared[MODEL_COLS], prepared["home_win"])
    return model