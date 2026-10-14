
"""Utilities for analyzing hypothetical NHL moneyline bets."""


def implied_probability(american_odds):
    """Convert American moneyline odds into implied probability."""
    odds = float(american_odds)

    if odds == 0:
        raise ValueError("American odds cannot be zero.")

    if odds > 0:
        return 100 / (odds + 100)

    return abs(odds) / (abs(odds) + 100)


def decimal_odds(american_odds):
    """Convert American odds into decimal odds."""
    odds = float(american_odds)

    if odds == 0:
        raise ValueError("American odds cannot be zero.")

    if odds > 0:
        return 1 + odds / 100

    return 1 + 100 / abs(odds)


def no_vig_probabilities(home_odds, away_odds):
    """Remove the market margin by normalizing implied probabilities."""
    home = implied_probability(home_odds)
    away = implied_probability(away_odds)

    total = home + away

    return {
        "home": home / total,
        "away": away / total,
        "overround": total - 1,
    }


def expected_value(win_probability, american_odds, stake=10):
    """Calculate theoretical net profit per hypothetical wager."""
    if not 0 <= win_probability <= 1:
        raise ValueError("Win probability must be between 0 and 1.")

    if stake <= 0:
        raise ValueError("Stake must be positive.")

    profit_if_win = stake * (decimal_odds(american_odds) - 1)
    loss_if_lose = stake

    return (
        win_probability * profit_if_win
        - (1 - win_probability) * loss_if_lose
    )


if __name__ == "__main__":
    # Example only: these are NOT actual sportsbook odds.
    model_home_probability = 0.656
    home_moneyline = -150
    away_moneyline = +130

    market = no_vig_probabilities(home_moneyline, away_moneyline)

    print("MODEL VS. MARKET — EXAMPLE")
    print("=" * 40)
    print(f"Model home win probability: {model_home_probability:.1%}")
    print(f"Raw market implied probability: {implied_probability(home_moneyline):.1%}")
    print(f"No-vig market probability: {market['home']:.1%}")
    print(f"Sportsbook margin: {market['overround']:.1%}")
    print(f"Model-market difference: {model_home_probability - market['home']:+.1%}")
    print(f"Theoretical EV per $10: ${expected_value(model_home_probability, home_moneyline):+.2f}")
