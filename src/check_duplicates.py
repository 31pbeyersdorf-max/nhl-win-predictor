import pandas as pd
from pipeline import load_games

games = load_games()

keys = ["gameId", "home_or_away"]

duplicates = games[
    games.duplicated(subset=keys, keep=False)
].copy()

print("Total duplicate rows:", len(duplicates))

if duplicates.empty:
    print("No duplicates found.")
else:
    print("\nExample duplicate records:")

    columns = [
        c for c in [
            "gameId",
            "home_or_away",
            "team",
            "gameDate",
            "goalsFor",
            "goalsAgainst"
        ]
        if c in duplicates.columns
    ]

    print(
        duplicates[columns]
        .sort_values(keys)
        .head(20)
        .to_string(index=False)
    )

    print("\nConflicting goalsFor values:")

    conflicts = (
        duplicates.groupby(keys)["goalsFor"]
        .nunique(dropna=False)
    )

    print(conflicts[conflicts > 1].head(20))

print("\nAffected seasons:")
print(
    duplicates.groupby(
        duplicates["gameDate"].dt.year
    ).size()
)

print("\nTeam combinations:")
print(
    duplicates.groupby(["gameId"])["team"]
    .apply(lambda x: ", ".join(sorted(x)))
    .head(20)
)

print("\nHome/away distribution:")
print(games["home_or_away"].value_counts())

print("\nARI games with duplicate sides:")
print(
    duplicates[duplicates["team"] == "ARI"][
        ["gameId", "team", "home_or_away", "goalsFor"]
    ].head(20).to_string(index=False)
)

print("\nARI home/away distribution by season:")

ari = games[games["team"] == "ARI"].copy()

ari["season"] = ari["gameDate"].dt.year

print(
    pd.crosstab(
        ari["season"],
        ari["home_or_away"]
    )
)

from pipeline import DATA_PATH

import pandas as pd

raw = pd.read_csv(DATA_PATH)

game_id = 2011020055

print("\nOriginal MoneyPuck records:")

original = raw[
    (pd.to_numeric(raw["gameId"], errors="coerce") == game_id)
    & (raw["situation"] == "all")
]

print(
    original[
        ["team", "gameId", "home_or_away",
         "goalsFor", "goalsAgainst"]
    ].to_string(index=False)
)

print("\nProcessed records:")

processed = games[games["gameId"] == game_id]

print(
    processed[
        ["team", "gameId", "home_or_away",
         "goalsFor", "goalsAgainst"]
    ].to_string(index=False)
)
