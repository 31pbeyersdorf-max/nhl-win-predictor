"""Sanity check: the refactored code should reproduce the notebook results."""
from sklearn.metrics import accuracy_score, log_loss
from pipeline import load_games, build_model_table, prepare, fit_model, MODEL_COLS

games = load_games()
table, _, _ = build_model_table(games)
table = table[table["season"] <= 2025]

train = prepare(table[table["season"] <= 2021])
test = prepare(table[table["season"] >= 2022])
model = fit_model(train)

p = model.predict_proba(test[MODEL_COLS])[:, 1]
print("Train games:", len(train), "| Test games:", len(test))
print("Accuracy:", round(accuracy_score(test["home_win"], p > 0.5), 4))
print("Log loss:", round(log_loss(test["home_win"], p), 4))
print("Expected from notebook: accuracy 0.5984, log loss 0.6623")