# NHL Predictive Analytics

An end-to-end NHL machine learning project built with Python that predicts game winners, evaluates historical performance, retrieves live sportsbook odds, and compares model probabilities against betting-market estimates.

The project combines expected-goals (xG) statistics, rolling team performance, Elo ratings, logistic regression, API integration, and an interactive Streamlit dashboard.

## Model Performance

The latest chronological holdout backtest evaluates 5,344 NHL games from the 2022–23 season through the early 2026–27 season.

| Metric | Result |
|---|---|
| Winner prediction accuracy | **58.78%** |
| Home-team baseline accuracy | 53.72% |
| Winner log loss | 0.6656 |
| Winner Brier score | 0.2367 |
| Games evaluated | 5,344 |
| Expected calibration error | 2.27% |

These results come from historical holdout evaluation. They do not establish profitability against sportsbook markets.

## Features

- **NHL winner predictions:** Predicts home and away win probabilities using historical team performance and Elo-based features.
- **Expected-goals analysis:** Uses rolling expected-goals and other team statistics to estimate relative strength.
- **Score estimates:** Generates preliminary expected home goals, away goals, total goals, and over-5.5 probabilities using statistical baselines.
- **Live sportsbook integration:** Retrieves NHL moneyline odds from The Odds API and stores timestamped sportsbook observations.
- **Model vs. Market:** Compares predicted probabilities with sportsbook-implied probabilities after adjusting for the bookmaker margin.
- **Probability calibration:** Evaluates whether predicted win probabilities match historical observed win rates.
- **Live prediction tracking:** Uses official NHL game results to grade timestamp-verified pregame predictions.
- **Interactive dashboard:** Visualizes predictions, historical backtests, sportsbook comparisons, and calibration results using Streamlit.

## Technologies

Python, pandas, NumPy, scikit-learn, SciPy, requests, python-dotenv, Streamlit, Git, and GitHub.

## Methodology

### 1. Data Collection

Historical game-level NHL team statistics are obtained from MoneyPuck.

The pipeline constructs a game-level dataset using home and away team observations.

### 2. Feature Engineering

The model uses historical team-performance features, including rolling 15-game statistics, opponent-adjusted expected goals, and Elo ratings.

Pregame features are designed to exclude information from the game being predicted.

### 3. Machine Learning

Logistic regression generates home-team win probabilities from engineered features.

Additional experiments investigate Elo-based approaches and alternative algorithms.

### 4. Historical Evaluation

The primary historical backtest trains the winner model on seasons before 2022 and evaluates it on later seasons.

Resolved shootout games are included in winner evaluation. Games without a verified winner are excluded.

### 5. Probability Calibration

Historical holdout predictions are grouped by predicted win probability and compared against actual outcomes.

The project also tests Platt scaling and isotonic regression using a chronological calibration/evaluation split.

Neither method demonstrated a meaningful improvement over the original model in the latest experiment, so the original probabilities are retained.

### 6. Sportsbook Market Comparison

The Odds API supplies NHL moneyline prices from multiple sportsbooks.

The project calculates:

- Implied win probabilities
- No-vig market probabilities
- Model-market probability differences
- Theoretical expected value for hypothetical stakes

Sportsbook observations include capture timestamps and are stored for later analysis.

### 7. Live Performance Tracking

Predictions made before scheduled game start times are evaluated against official NHL final results.

A separate paper-performance tracker compares model and market predictions on completed games and calculates hypothetical returns.

## Project Structure

```text
nhl-win-predictor/
├── app.py
├── README.md
├── requirements.txt
├── src/
│   ├── pipeline.py
│   ├── predict_upcoming.py
│   ├── update_data.py
│   ├── backtest_models.py
│   ├── calibrate_model.py
│   ├── test_recalibration.py
│   ├── fetch_market_odds.py
│   ├── compare_market.py
│   ├── track_predictions.py
│   └── track_market_performance.py
├── notebooks/
├── figures/
├── data/           # Local data, excluded from Git
└── predictions/    # Generated results, excluded from Git
```

## Running the Project

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Configure the odds API

Obtain an API key from [The Odds API](https://the-odds-api.com/).

Create a `.env` file in the project root:

```text
ODDS_API_KEY=your_api_key_here
```

The `.env` file is excluded from version control.

### Prepare historical data

Download the appropriate MoneyPuck team data and place it in the local `data/` directory.

Some historical winner-label corrections also require the project's NHL winner-resolution data.

### Run the historical backtest

```bash
python src/backtest_models.py
python src/calibrate_model.py
python src/test_recalibration.py
```

### Generate and compare predictions

After preparing the required local data:

```bash
python src/predict_upcoming.py
python src/fetch_market_odds.py
python src/compare_market.py
```

### Track completed games

```bash
python src/track_predictions.py
python src/track_market_performance.py
```

### Launch the dashboard

```bash
python -m streamlit run app.py
```

The dashboard displays available predictions, historical performance, sportsbook comparisons, and probability calibration.

## Limitations

- No confirmed real-world betting advantage or profitability.
- Market comparisons depend on sportsbook availability and timestamped price collection.
- Historical calibration does not guarantee that the current production model is calibrated.
- Goalie assignments, injuries, lineup changes, and other contextual factors are not comprehensively modeled.
- Score and total-goal estimates are preliminary statistical baselines.
- Historical data files and generated prediction outputs are not included in the repository.
- The complete reproduction process requires the relevant historical data and winner-resolution files.

## Future Improvements

- Automated daily data updates and odds collection
- Larger prospective model-vs-market evaluation
- Probability calibration monitoring across NHL seasons
- Goalie, injury, and lineup features
- Expanded Streamlit visualizations
- More complete reproducibility and automated testing

## Data Sources

- [MoneyPuck](https://moneypuck.com/data.htm) — Historical NHL team and game statistics
- [NHL](https://www.nhl.com/) — Official game results and scheduling data
- [The Odds API](https://the-odds-api.com/) — Sportsbook moneyline data

## Disclaimer

This project is for educational, statistical research, and portfolio demonstration purposes. Theoretical expected values and paper returns are not evidence of guaranteed profits or recommendations to place wagers.