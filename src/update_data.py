
"""Safely download the latest MoneyPuck team data."""

from pathlib import Path
from datetime import datetime, timezone
import requests

URL = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"

DATA_FILE = Path("data/all_teams.csv")
TEMP_FILE = Path("data/all_teams_download.tmp")
BACKUP_FILE = Path("data/all_teams_backup.csv")


def main():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading latest MoneyPuck data...")

    try:
        with requests.get(URL, stream=True, timeout=120) as response:
            response.raise_for_status()

            with open(TEMP_FILE, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        size_mb = TEMP_FILE.stat().st_size / (1024 * 1024)

        if size_mb < 50:
            raise ValueError(
                f"Downloaded file is unexpectedly small: {size_mb:.1f} MB"
            )

        with open(TEMP_FILE, "rb") as file:
            header = file.readline().decode("utf-8-sig").strip()

        required = {"gameId", "gameDate", "team", "situation"}

        if not required.issubset(set(header.split(","))):
            raise ValueError("Downloaded CSV is missing required columns.")

        if DATA_FILE.exists():
            DATA_FILE.replace(BACKUP_FILE)

        TEMP_FILE.replace(DATA_FILE)

        print(f"Download successful: {size_mb:.1f} MB")
        print(f"Updated at: {datetime.now(timezone.utc).isoformat()}")
        print(f"Saved to: {DATA_FILE}")
        print(f"Previous version backed up to: {BACKUP_FILE}")

    except Exception as error:
        print(f"Update failed: {error}")
        if TEMP_FILE.exists():
            TEMP_FILE.unlink()
        print("Check your existing data before continuing.")


if __name__ == "__main__":
    main()
