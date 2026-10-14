

"""Download official NHL winners using concurrent requests."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from pipeline import load_games

OUTPUT = Path("data/nhl_tied_game_winners.csv")
API = "https://api-web.nhle.com/v1/gamecenter/{}/landing"

MAX_WORKERS = 5


def fetch_game(game_id):
    try:
        response = requests.get(
            API.format(game_id),
            timeout=15
        )
        response.raise_for_status()

        data = response.json()

        if data.get("gameState") not in ("FINAL", "OFF"):
            return None

        home_score = data.get("homeTeam", {}).get("score")
        away_score = data.get("awayTeam", {}).get("score")

        if home_score is None or away_score is None:
            return None

        if home_score == away_score:
            return None

        return {
            "gameId": str(game_id),
            "home_win": int(home_score > away_score),
            "home_score": home_score,
            "away_score": away_score,
        }

    except requests.RequestException as error:
        print(f"Error fetching {game_id}: {error}")
        return None


def main():
    games = load_games()

    home = games[
        games["home_or_away"] == "HOME"
    ].copy()

    tied = home[
        home["goal_diff"] == 0
    ].copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        saved = pd.read_csv(
            OUTPUT, dtype={"gameId": str}
        )
        records = saved.to_dict("records")
        completed = set(saved["gameId"])
    else:
        records = []
        completed = set()

    ids = [
        str(game_id)
        for game_id in tied["gameId"]
        if str(game_id) not in completed
    ]

    print(f"Games remaining: {len(ids)}")

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(fetch_game, game_id): game_id
            for game_id in ids
        }

        for count, future in enumerate(
            as_completed(futures), 1
        ):
            result = future.result()

            if result is not None:
                records.append(result)

            if count % 50 == 0:
                print(
                    f"Processed {count}/{len(ids)} games"
                )

                pd.DataFrame(records).to_csv(
                    OUTPUT, index=False
                )

    if records:
        pd.DataFrame(records).to_csv(
            OUTPUT, index=False
        )

    print(f"\nSaved {len(records)} official results.")
    print(f"File: {OUTPUT}")


if __name__ == "__main__":
    main()
