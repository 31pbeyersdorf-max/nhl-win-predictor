
"""Fetch and preserve NHL moneyline odds for paper analysis."""

import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("ODDS_API_KEY")
PREDICTIONS = ROOT / "predictions/predictions.csv"
OUTPUT = ROOT / "data/market_odds.csv"

URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"

TEAM_NAMES = {
    "Anaheim Ducks": "ANA",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Mammoth": "UTA",
    "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}

COLUMNS = [
    "game_id", "home_odds", "away_odds",
    "recorded_at_utc", "sportsbook", "odds_type",
    "source", "bookmaker_last_update_utc",
    "api_event_id",
]


def normalize_name(name):
    return unicodedata.normalize("NFC", name.strip())


def main():
    if not API_KEY:
        raise RuntimeError("Missing ODDS_API_KEY in .env")

    predictions = pd.read_csv(
        PREDICTIONS, dtype={"game_id": str}
    )

    predictions["game_start_utc"] = pd.to_datetime(
        predictions["game_start_utc"],
        utc=True, errors="coerce"
    )

    predictions["predicted_at_utc"] = pd.to_datetime(
        predictions["predicted_at_utc"],
        utc=True, errors="coerce"
    )

    predictions = predictions[
        predictions["game_start_utc"].notna()
        & predictions["predicted_at_utc"].notna()
        & (predictions["predicted_at_utc"] < predictions["game_start_utc"])
    ].copy()

    predictions = predictions.drop_duplicates("game_id")

    response = requests.get(
        URL,
        params={
            "apiKey": API_KEY,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
        },
        timeout=30,
    )
    response.raise_for_status()

    fetched_at = datetime.now(timezone.utc)
    rows = []
    unmatched = []

    for event in response.json():
        home_name = normalize_name(event["home_team"])
        away_name = normalize_name(event["away_team"])

        home_code = TEAM_NAMES.get(home_name)
        away_code = TEAM_NAMES.get(away_name)

        event_start = pd.to_datetime(
            event["commence_time"], utc=True
        )

        if not home_code or not away_code:
            unmatched.append((home_name, away_name))
            continue

        matches = predictions[
            (predictions["home_team"] == home_code)
            & (predictions["away_team"] == away_code)
            & (
                abs(
                    predictions["game_start_utc"]
                    - event_start
                ) <= pd.Timedelta(hours=3)
            )
        ]

        if len(matches) != 1:
            unmatched.append((away_name, home_name))
            continue

        prediction = matches.iloc[0]

        if fetched_at >= event_start.to_pydatetime():
            continue

        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                prices = {
                    normalize_name(o["name"]): o["price"]
                    for o in market.get("outcomes", [])
                }

                if home_name not in prices or away_name not in prices:
                    continue

                rows.append({
                    "game_id": prediction["game_id"],
                    "home_odds": prices[home_name],
                    "away_odds": prices[away_name],
                    "recorded_at_utc": fetched_at.isoformat(),
                    "sportsbook": book["key"],
                    "odds_type": "moneyline_full_game",
                    "source": "the_odds_api",
                    "bookmaker_last_update_utc": book.get("last_update"),
                    "api_event_id": event["id"],
                })

    new_data = pd.DataFrame(rows, columns=COLUMNS)

    if OUTPUT.exists():
        old_data = pd.read_csv(OUTPUT, dtype={"game_id": str})
        combined = pd.concat(
            [old_data, new_data], ignore_index=True
        )
    else:
        combined = new_data

    combined.to_csv(OUTPUT, index=False)

    print(f"Fetched {len(response.json())} NHL events.")
    print(f"Saved {len(new_data)} new sportsbook observations.")
    print(f"Total stored odds observations: {len(combined)}")
    print(
        "API credits remaining:",
        response.headers.get("x-requests-remaining", "Unknown")
    )

    if unmatched:
        print("Unmatched games:")
        for game in unmatched:
            print(" ", game)


if __name__ == "__main__":
    main()
