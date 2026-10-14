"""Chronological holdout backtest for NHL winner and Poisson goal models.
Run from repository root: python src/backtest_models.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error

from pipeline import MODEL_COLS, load_games, build_model_table, prepare, fit_model
from score_model import fit_score_models

TEST_START_SEASON = 2022  # season starting in 2022 (2022-23)
OUT_DIR = Path('predictions')


def main():
    games = load_games()
    # Never arbitrarily discard conflicting game records: detect data issues first.
    counts = games.groupby(['gameId', 'home_or_away']).size()
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError(
            f'{len(bad)} game/side groups do not have exactly one row. '
            f'Inspect the source data before backtesting. Examples:\n{bad.head(12)}'
        )

    table, _, _ = build_model_table(games)
    if table['gameId'].duplicated().any():
        raise ValueError('Feature table contains duplicate gameId values.')

    home = games.loc[games.home_or_away == 'HOME', ['gameId', 'goalsFor']].rename(columns={'goalsFor': 'home_goals'})
    away = games.loc[games.home_or_away == 'AWAY', ['gameId', 'goalsFor']].rename(columns={'goalsFor': 'away_goals'})
    goal_table = table.merge(home, on='gameId', validate='one_to_one').merge(away, on='gameId', validate='one_to_one')
    goal_table = goal_table.dropna(subset=MODEL_COLS + ['home_goals', 'away_goals']).copy()
    goal_table['season'] = goal_table['season'].astype(int)
    goal_train = goal_table[goal_table.season < TEST_START_SEASON]
    goal_test = goal_table[goal_table.season >= TEST_START_SEASON].copy()


    win_table = prepare(table)
    win_table['season'] = win_table['season'].astype(int)

    win_train = win_table[
        win_table.season < TEST_START_SEASON
]

    win_test = win_table[
        win_table.season >= TEST_START_SEASON
].copy()


    if min(len(goal_train), len(goal_test), len(win_train), len(win_test)) == 0:
        raise ValueError('No training or testing data. Check TEST_START_SEASON and dataset.')

    winner = fit_model(win_train)
    scorers = fit_score_models(goal_train)
    win_test['home_win_prob'] = winner.predict_proba(win_test[MODEL_COLS])[:, 1]
    win_test['predicted_home_win'] = (win_test.home_win_prob >= .5).astype(int)

    goal_test['expected_home_goals'] = scorers['home'].predict(goal_test[MODEL_COLS])
    goal_test['expected_away_goals'] = scorers['away'].predict(goal_test[MODEL_COLS])
    goal_test['expected_total_goals'] = goal_test.expected_home_goals + goal_test.expected_away_goals
    goal_test['actual_total_goals'] = goal_test.home_goals + goal_test.away_goals
    goal_test['over_5_5_prob'] = poisson.sf(5, goal_test.expected_total_goals)
    goal_test['actual_over_5_5'] = (goal_test.actual_total_goals >= 6).astype(int)


    # Winner evaluation now includes resolved shootout games.
    # Unresolved games are excluded from winner evaluation.

    # Ensure game IDs use the same data type before merging.
    goal_test = goal_test.copy()
    win_test = win_test.copy()

    goal_test["gameId"] = goal_test["gameId"].astype(str).str.strip()
    win_test["gameId"] = win_test["gameId"].astype(str).str.strip()

    results = goal_test[['gameId', 'season', 'gameDate', 'home_team', 'away_team',
                         'home_goals', 'away_goals', 'expected_home_goals',
                         'expected_away_goals', 'actual_total_goals', 'expected_total_goals',
                         'over_5_5_prob', 'actual_over_5_5']].merge(
        win_test[['gameId', 'home_win', 'home_win_prob', 'predicted_home_win']],
        on='gameId', how='left', validate='one_to_one'
    )

    results['winner_correct'] = np.where(
        results.home_win.notna(),
        (results.predicted_home_win == results.home_win).astype(float),
        np.nan
    )

    results['home_goal_abs_error'] = abs(results.expected_home_goals - results.home_goals)
    results['away_goal_abs_error'] = abs(results.expected_away_goals - results.away_goals)
    results['total_goal_abs_error'] = abs(results.expected_total_goals - results.actual_total_goals)

    OUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUT_DIR / 'backtest_predictions.csv', index=False)


    metrics = []
    for season, subset in [('All test seasons', results), *[(str(s), x) for s, x in results.groupby('season')]]:
        w = subset.dropna(subset=['home_win', 'home_win_prob'])
        metrics.append({
            'season': season,
            'games': len(subset), 'winner_games': len(w),
            'winner_accuracy': accuracy_score(w.home_win, w.predicted_home_win) if len(w) else np.nan,
            'home_baseline_accuracy': w.home_win.mean() if len(w) else np.nan,
            'winner_log_loss': log_loss(w.home_win, w.home_win_prob, labels=[0, 1]) if len(w) else np.nan,
            'winner_brier': brier_score_loss(w.home_win, w.home_win_prob) if len(w) else np.nan,
            'home_goal_mae': subset.home_goal_abs_error.mean(),
            'away_goal_mae': subset.away_goal_abs_error.mean(),
            'total_goal_mae': subset.total_goal_abs_error.mean(),
            'over_5_5_brier': brier_score_loss(subset.actual_over_5_5, subset.over_5_5_prob),
        })
    summary = pd.DataFrame(metrics)
    summary.to_csv(OUT_DIR / 'backtest_metrics.csv', index=False)
    print('\nHoldout backtest (no test-season model fitting):')
    print(summary.round(4).to_string(index=False))
    print('\nSaved predictions/backtest_predictions.csv and predictions/backtest_metrics.csv')
    print('Note: scoring targets are MoneyPuck recorded final goals; over/under is a Poisson baseline.')


if __name__ == '__main__':
    main()
