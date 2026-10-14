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
WINDOW = 15      # games in each rolling average
MIN_GAMES = 7   # need at least this many earlier games to make features




FEATURES = (
    [s + "_avg10" for s in STATS]
    + ["rest_days"]
    + ["adjusted_xgf_avg15", "adjusted_xga_avg15"]
)



MODEL_COLS = (["home_" + f for f in FEATURES]
              + ["away_" + f for f in FEATURES]
              + ["elo_logit"])

ELO_K = 10
ELO_HOME_ADV = 0
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

    # Remove games with invalid home/away assignments.
    # A valid NHL game must have exactly one HOME and one AWAY row.
    side_counts = (
        g.groupby(["gameId", "home_or_away"])
        .size()
        .unstack(fill_value=0)
    )

    valid_ids = side_counts.index[
        (side_counts.get("HOME", 0) == 1)
        & (side_counts.get("AWAY", 0) == 1)
    ]

    removed = g["gameId"].nunique() - len(valid_ids)

    if removed:
        print(f"Excluded {removed} games with invalid home/away records.")

    g = g[g["gameId"].isin(valid_ids)].copy()

    g["goal_diff"] = g["goalsFor"] - g["goalsAgainst"]

    return g.sort_values(
        ["team", "gameDate"]
    ).reset_index(drop=True)

def add_recent_form(games, window=5):
    """Calculate team form using only previous games."""
    games = games.sort_values(
        ["team", "gameDate", "gameId"]
    ).copy()

    recent_stats = [
        "goalsFor",
        "goalsAgainst"
    ]

    for stat in recent_stats:
        games[f"last{window}_{stat}"] = (
            games.groupby("team")[stat]
            .transform(
                lambda x: x.shift(1)
                .rolling(window, min_periods=3)
                .mean()
            )
        )

    return games




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

def add_opponent_adjusted_features(games):
    """Measure xG performance against opponents' pregame strength."""

    games = games.sort_values(
        ["gameDate", "gameId", "team"]
    ).copy()

    # Pregame defensive and offensive averages.
    # These only use games played BEFORE the current matchup.
    games["pregame_xga"] = (
        games.groupby("team")["xGoalsAgainst"]
        .transform(
            lambda s: s.shift(1).rolling(
                WINDOW, min_periods=MIN_GAMES
            ).mean()
        )
    )

    games["pregame_xgf"] = (
        games.groupby("team")["xGoalsFor"]
        .transform(
            lambda s: s.shift(1).rolling(
                WINDOW, min_periods=MIN_GAMES
            ).mean()
        )
    )

    # Attach the opponent's PRE-game averages to each team row.
    opponent = games[
        ["gameId", "team", "pregame_xga", "pregame_xgf"]
    ].rename(
        columns={
            "team": "opponent",
            "pregame_xga": "opponent_pregame_xga",
            "pregame_xgf": "opponent_pregame_xgf",
        }
    )

    games = games.merge(
        opponent,
        on="gameId",
        suffixes=("", "_opponent")
    )

    # Keep only the OTHER team from the matchup.
    games = games[
        games["team"] != games["opponent"]
    ].copy()

    # How well did the team perform relative to the opponent?
    games["adjusted_xgf"] = (
        games["xGoalsFor"] - games["opponent_pregame_xga"]
    )

    games["adjusted_xga"] = (
        games["xGoalsAgainst"] - games["opponent_pregame_xgf"]
    )

    games = games.sort_values(
        ["team", "gameDate", "gameId"]
    )

    # Roll previous-game adjustments into pregame features.
    for stat in ["adjusted_xgf", "adjusted_xga"]:
        games[stat + "_avg15"] = (
            games.groupby("team")[stat]
            .transform(
                lambda s: s.shift(1).rolling(
                    WINDOW, min_periods=MIN_GAMES
                ).mean()
            )
        )

    return games.drop(
        columns=[
            "opponent",
            "pregame_xga",
            "pregame_xgf",
            "opponent_pregame_xga",
            "opponent_pregame_xgf",
        ]
    )



def elo_expected(r_home, r_away):
    return 1 / (1 + 10 ** (-((r_home + ELO_HOME_ADV) - r_away) / 400))


def regress_to_mean(ratings):
    """New season: pull every team 25% of the way back to average."""
    for t in ratings:
        ratings[t] = 0.75 * ratings[t] + 0.25 * ELO_START



def run_elo(games):
    """Calculate pregame Elo ratings using outright NHL winners."""

    home = (
        games[games["home_or_away"] == "HOME"]
        [["gameId", "gameDate", "season", "team", "goal_diff"]]
        .rename(columns={"team": "home_team"})
    )

    away = (
        games[games["home_or_away"] == "AWAY"]
        [["gameId", "team"]]
        .rename(columns={"team": "away_team"})
    )

    g = (
        home.merge(away, on="gameId")
        .sort_values(["gameDate", "gameId"])
        .reset_index(drop=True)
    )

    # Official winners for tied-score games.
    winners = pd.read_csv(
        "data/nhl_tied_game_winners.csv",
        dtype={"gameId": str}
    )

    winners["gameId"] = winners["gameId"].str.strip()

    if winners["gameId"].duplicated().any():
        raise ValueError("Duplicate game IDs in winner lookup.")

    winner_lookup = winners.set_index("gameId")["home_win"]

    ratings = {}
    probs = []
    last_season = None

    for row in g.itertuples():
        if last_season is not None and row.season != last_season:
            regress_to_mean(ratings)

        last_season = row.season

        rh = ratings.get(row.home_team, ELO_START)
        ra = ratings.get(row.away_team, ELO_START)

        p = elo_expected(rh, ra)

        # Store probability BEFORE updating Elo.
        probs.append(p)

        if row.goal_diff > 0:
            result = 1.0
        elif row.goal_diff < 0:
            result = 0.0
        else:
            game_id = str(row.gameId).strip()
            official_winner = winner_lookup.get(game_id)

            if pd.notna(official_winner):
                result = float(official_winner)
            else:
                # Unknown winner: don't update either rating.
                result = None

        if result is not None:
            ratings[row.home_team] = rh + ELO_K * (result - p)
            ratings[row.away_team] = ra - ELO_K * (result - p)

    g["elo_prob"] = probs

    return g[["gameId", "elo_prob"]], ratings, last_season

def build_model_table(games):
    """One row per game: home features, away features, Elo."""


    feat = add_rolling_features(games)
    feat = add_opponent_adjusted_features(feat)



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
    """Create outright winner labels, including resolved shootouts."""
    t = table.copy()

    # Non-tied games already have a clear winner.
    t["home_win"] = np.where(
        t["goal_diff"] > 0, 1,
        np.where(t["goal_diff"] < 0, 0, np.nan)
    )

    # Load official results for games tied in MoneyPuck.
    winners_path = "data/nhl_tied_game_winners.csv"
    winners = pd.read_csv(
        winners_path,
        dtype={"gameId": str}
    )

    winners["gameId"] = winners["gameId"].str.strip()
    winners["home_win"] = pd.to_numeric(
        winners["home_win"], errors="coerce"
    )

    if winners["gameId"].duplicated().any():
        raise ValueError("Duplicate game IDs in winner lookup.")

    # Match IDs consistently regardless of their original types.
    t["gameId"] = t["gameId"].astype(str).str.strip()

    winner_lookup = winners.set_index("gameId")["home_win"]

    tied = t["goal_diff"] == 0
    t.loc[tied, "home_win"] = (
        t.loc[tied, "gameId"].map(winner_lookup)
    )

    unresolved = t["home_win"].isna().sum()
    print(f"Games without resolved winner: {unresolved}")

    t = t.dropna(subset=["home_win"] + MODEL_COLS).copy()
    t["home_win"] = t["home_win"].astype(int)

    return t



def fit_model(prepared):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(prepared[MODEL_COLS], prepared["home_win"])
    return model