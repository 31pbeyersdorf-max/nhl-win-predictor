
import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path("dashboard_data")


def get_last_updated():
    timestamp_file = DATA_DIR / "last_updated.txt"

    if timestamp_file.exists():
        return timestamp_file.read_text().strip()

    return "Not available"


st.set_page_config(
    page_title="NHL Predictive Analytics",
    page_icon="🏒",
    layout="wide"
)

st.title("🏒 NHL Predictive Analytics")
st.caption("Machine learning predictions powered by Python")

st.info(
    "NHL predictions are refreshed daily through GitHub Actions. "
    "Verified live performance updates as completed games become available. "
    "Historical backtest metrics and sportsbook odds are separate snapshots."
)

st.caption(f"🕒 Predictions last updated (UTC): {get_last_updated()}")

df = pd.read_csv(DATA_DIR / "predictions.csv")


# Keep only rows containing the new score predictions
df = df.dropna(
    subset=["expected_home_goals", "expected_away_goals"]
).copy()

df["game_date"] = pd.to_datetime(df["game_date"])

# Latest available prediction per game
df = (
    df.sort_values("predicted_on")
    .drop_duplicates(subset=["game_id"], keep="last")
)

st.sidebar.header("Filters")

from datetime import datetime
from zoneinfo import ZoneInfo

dates = sorted(df["game_date"].dt.date.unique())

today = datetime.now(ZoneInfo("America/New_York")).date()

upcoming_dates = [date for date in dates if date >= today]

if upcoming_dates:
    default_date = upcoming_dates[0]
else:
    default_date = dates[-1]

selected_date = st.sidebar.selectbox(
    "Select game date",
    dates,
    index=dates.index(default_date)
)

games = df[df["game_date"].dt.date == selected_date]

st.subheader(f"Predictions for {selected_date}")

if games.empty:
    st.info("No predictions available for this date.")

for _, game in games.iterrows():
    away = game["away_team"]
    home = game["home_team"]

    with st.container(border=True):
        st.subheader(f"{away} @ {home}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                f"{away} Expected Goals",
                f"{game['expected_away_goals']:.2f}"
            )

        with col2:
            st.metric(
                "Most Likely Baseline Score",
                f"{int(game['predicted_away_score'])} - "
                f"{int(game['predicted_home_score'])}"
            )

        with col3:
            st.metric(
                f"{home} Expected Goals",
                f"{game['expected_home_goals']:.2f}"
            )

        st.write("**Win Probabilities (existing model)**")

        home_prob = float(game["home_win_prob"])
        st.progress(home_prob)

        st.write(
            f"{away}: {(1-home_prob):.1%} | "
            f"{home}: {home_prob:.1%}"
        )

        st.write(
            f"**Expected Total Goals:** "
            f"{game['expected_total_goals']:.2f}"
        )

        st.write(
            f"**Over 5.5 Probability (Poisson baseline):** "
            f"{game['over_5_5_prob']:.1%}"
        )

        st.caption(
            "Score predictions are preliminary statistical estimates, "
            "not validated NHL final-score probabilities."
        )

# Historical model performance
st.divider()
st.header("📊 Historical Model Performance")

try:
    metrics = pd.read_csv(DATA_DIR / "backtest_metrics.csv")

    overall = metrics[
        metrics["season"].astype(str) == "All test seasons"
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Winner Accuracy",
        f"{overall['winner_accuracy']:.1%}"
    )

    col2.metric(
        "Home Baseline",
        f"{overall['home_baseline_accuracy']:.1%}"
    )

    col3.metric(
        "Winner Log Loss",
        f"{overall['winner_log_loss']:.4f}"
    )

    col4.metric(
        "Total Goals MAE",
        f"{overall['total_goal_mae']:.2f}"
    )

    st.subheader("Winner Accuracy by Season")

    seasons = metrics[
        metrics["season"].astype(str) != "All test seasons"
    ].copy()

    seasons["season"] = seasons["season"].astype(str)

    accuracy_chart = seasons.set_index("season")[
        ["winner_accuracy", "home_baseline_accuracy"]
    ] * 100

    accuracy_chart.columns = [
        "NHL Model Accuracy (%)",
        "Home Team Baseline (%)"
    ]

    st.line_chart(accuracy_chart)

    st.subheader("Goal Prediction Error")

    goal_chart = seasons.set_index("season")[
        ["home_goal_mae", "away_goal_mae", "total_goal_mae"]
    ]

    goal_chart.columns = [
        "Home Goals MAE",
        "Away Goals MAE",
        "Total Goals MAE"
    ]

    st.line_chart(goal_chart)

    st.subheader("Detailed Backtest Results")
    st.dataframe(metrics, use_container_width=True)

except (FileNotFoundError, KeyError, IndexError) as error:
    st.warning(f"Backtest results unavailable: {error}")

from pathlib import Path

st.divider()
st.header("Live Prediction Tracking")

metrics_file = DATA_DIR / "verified_live_metrics.csv"
results_file = DATA_DIR / "verified_live_metrics.csv"

if metrics_file.exists():
    metrics = pd.read_csv(metrics_file).iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Completed Predictions",
        int(metrics["completed_games"])
    )
    col2.metric(
        "Winner Accuracy",
        f"{metrics['accuracy']:.1%}"
    )
    col3.metric(
        "Brier Score",
        f"{metrics['brier']:.4f}"
    )
else:
    st.info(
        "Five verified pregame predictions are logged. "
        "Accuracy will appear once games finish "
        "and the tracker runs."
    )

if results_file.exists():
    st.subheader("Completed Game Results")
    results = pd.read_csv(results_file)
    st.dataframe(results, use_container_width=True)
if results_file.exists():
    st.subheader("Completed Game Results")
    results = pd.read_csv(results_file)
    st.dataframe(results, use_container_width=True)



# Model vs. sportsbook market
st.divider()
st.header("💰 Model vs. Sportsbook Market")

market_file = DATA_DIR / "market_comparison.csv"

if market_file.exists():
    market = pd.read_csv(market_file)

    if market.empty:
        st.info("No sportsbook comparisons available yet.")

    else:
        st.caption(
            "Live pregame moneyline comparisons. "
            "Expected values are theoretical, not proven returns."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Games Compared",
            market["game_id"].nunique()
        )

        col2.metric(
            "Sportsbooks",
            market["sportsbook"].nunique()
        )

        col3.metric(
            "Price Comparisons",
            len(market)
        )

        # Interactive filters
        st.subheader("Explore Market Prices")

        selected_book = st.selectbox(
            "Sportsbook",
            ["All Sportsbooks"]
            + sorted(market["sportsbook"].unique().tolist())
        )

        if selected_book != "All Sportsbooks":
            filtered = market[
                market["sportsbook"] == selected_book
            ].copy()
        else:
            filtered = market.copy()

        filtered["matchup"] = (
            filtered["away_team"]
            + " @ "
            + filtered["home_team"]
        )

        selected_matchup = st.selectbox(
            "Matchup",
            ["All Games"]
            + sorted(filtered["matchup"].unique().tolist())
        )

        if selected_matchup != "All Games":
            filtered = filtered[
                filtered["matchup"] == selected_matchup
            ].copy()

        # Convert probability difference to percentage points
        filtered["Model Edge (pp)"] = (
            filtered["home_model_edge"] * 100
        )

        display = filtered[
            [
                "matchup",
                "sportsbook",
                "home_odds",
                "away_odds",
                "home_win_prob",
                "market_home_no_vig",
                "Model Edge (pp)",
                "home_ev_per_10",
                "away_ev_per_10",
            ]
        ].copy()

        display.columns = [
            "Matchup",
            "Sportsbook",
            "Home Moneyline",
            "Away Moneyline",
            "Model Home Win %",
            "Market Home Win %",
            "Model Edge (pp)",
            "Home EV / $10",
            "Away EV / $10",
        ]

        display["Model Home Win %"] *= 100
        display["Market Home Win %"] *= 100

        st.dataframe(
            display.style.format(
                {
                    "Model Home Win %": "{:.1f}%",
                    "Market Home Win %": "{:.1f}%",
                    "Model Edge (pp)": "{:+.1f}",
                    "Home EV / $10": "${:+.2f}",
                    "Away EV / $10": "${:+.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Visualize the model against the market
        if not filtered.empty:
            st.subheader("Model Probability vs. Market")

            chart_data = (
                filtered.groupby("matchup")[
                    ["home_win_prob", "market_home_no_vig"]
                ]
                .mean()
                * 100
            )

            chart_data.columns = [
                "Model Home Win %",
                "Market Home Win %",
            ]

            st.bar_chart(chart_data)

        st.caption(
            "Market probabilities exclude the estimated "
            "sportsbook margin. Model edge is the difference "
            "between model and no-vig probabilities. "
            "A positive theoretical EV does not guarantee "
            "a profitable betting strategy."
        )

else:
    st.info(
        "No market comparison file yet. "
        "Run python src/compare_market.py first."
    )

# Probability calibration
st.divider()
st.header("🎯 Model Probability Calibration")

calibration_file = DATA_DIR / "calibration_results.csv"
summary_file = DATA_DIR / "calibration_summary.csv"

if calibration_file.exists() and summary_file.exists():
    calibration = pd.read_csv(calibration_file)
    summary = pd.read_csv(summary_file).iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Evaluated Games",
        f"{int(summary['evaluated_games']):,}"
    )
    col2.metric(
        "Expected Calibration Error",
        f"{summary['expected_calibration_error']:.2%}"
    )
    col3.metric(
        "Brier Score",
        f"{summary['brier_score']:.4f}"
    )

    chart = calibration[
        [
            "bucket",
            "average_predicted_probability",
            "actual_home_win_rate",
        ]
    ].copy()

    chart = chart.set_index("bucket") * 100
    chart.columns = [
        "Predicted Win Probability (%)",
        "Actual Win Rate (%)",
    ]

    st.subheader("Predicted vs. Actual Win Rates")
    st.bar_chart(chart)

    st.subheader("Detailed Calibration Results")
    st.dataframe(
        calibration,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Calibration is evaluated on historical holdout "
        "predictions. These results do not establish "
        "profitable betting performance."
    )
else:
    st.info("Run python src/calibrate_model.py first.")
