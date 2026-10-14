
import os
import requests
from dotenv import load_dotenv

from pathlib import Path

load_dotenv(
    Path(__file__).resolve().parent.parent / ".env"
)

api_key = os.getenv("ODDS_API_KEY")

if not api_key:
    raise RuntimeError("Missing ODDS_API_KEY in .env")

url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"

params = {
    "apiKey": api_key,
    "regions": "us",
    "markets": "h2h",
    "oddsFormat": "american",
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

games = response.json()

print(f"API connection successful!")
print(f"NHL games with available odds: {len(games)}")
print(
    "API credits remaining:",
    response.headers.get("x-requests-remaining", "Unknown")
)

for game in games[:5]:
    print(
        game["away_team"],
        "@",
        game["home_team"],
        "|",
        game["commence_time"]
    )
    import json

if games:
    example = games[0]

    print("\nEXAMPLE GAME DATA")
    print("Game ID:", example.get("id"))
    print("Home:", example.get("home_team"))
    print("Away:", example.get("away_team"))
    print("Start:", example.get("commence_time"))

    print("\nAVAILABLE SPORTSBOOKS")
    for bookmaker in example.get("bookmakers", []):
        print(
            bookmaker.get("key"),
            "|",
            bookmaker.get("title"),
            "|",
            bookmaker.get("last_update"),
        )

        for market in bookmaker.get("markets", []):
            if market.get("key") == "h2h":
                print(
                    json.dumps(
                        market.get("outcomes", []),
                        indent=2,
                    )
                )
                break
