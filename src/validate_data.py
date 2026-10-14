
"""Check NHL historical data quality and dates."""

import pandas as pd
from pipeline import load_games

games = load_games()

today = pd.Timestamp.now().normalize()

print("\nDATA VALIDATION REPORT")
print("=" * 45)

print("\nLatest game date:")
print(games["gameDate"].max())

future = games[games["gameDate"] > today]

print("\nFuture-dated rows:", len(future))
print("Future-dated games:", future["gameId"].nunique())

if not future.empty:
    print("\nExamples of future-dated records:")
    print(
        future[
            ["gameId", "gameDate", "team", "goalsFor"]
        ].head(15).to_string(index=False)
    )

print("\nGame counts by season:")
print(
    games.groupby("season")["gameId"]
    .nunique()
    .tail(8)
)

print("\nMissing goals:")
print(games[["goalsFor", "goalsAgainst"]].isna().sum())

print("\nLatest 10 games:")
print(
    games.sort_values("gameDate", ascending=False)[
        ["gameDate", "gameId", "team",
         "home_or_away", "goalsFor"]
    ].head(10).to_string(index=False)
)
