# ============================================================
#  IPL SCORE PREDICTOR — Beginner Friendly
#  Dataset: ipl_2026_deliveries.csv
#  Models: Linear Regression → Random Forest
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# STEP 1: Load the Data
# ─────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading Data")
print("=" * 50)

df = pd.read_csv("ipl_2026_deliveries.csv")
print(f"Total deliveries: {len(df)}")
print(f"Matches: {df['match_id'].nunique()}")
print(f"Teams: {sorted(df['batting_team'].unique())}")
print(df.head(3))


# ─────────────────────────────────────────
# STEP 2: Feature Engineering
# Build snapshot of match state at each over
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Feature Engineering")
print("=" * 50)

# Total runs per ball = bat runs + extras
df['total_runs'] = df['runs_of_bat'] + df['extras']

# A wicket happened if wicket_type is filled
df['is_wicket'] = df['wicket_type'].notna().astype(int)

# Extract over number (e.g. 3.4 → over 3)
df['over_num'] = df['over'].apply(lambda x: int(str(x).split('.')[0]))

# Group by match + innings + over → get stats at end of each over
over_stats = df.groupby(['match_id', 'innings', 'batting_team', 'bowling_team', 'venue', 'over_num']).agg(
    runs_this_over=('total_runs', 'sum'),
    wickets_this_over=('is_wicket', 'sum')
).reset_index()

# Cumulative runs and wickets up to this over
over_stats = over_stats.sort_values(['match_id', 'innings', 'over_num'])
over_stats['cumulative_runs'] = over_stats.groupby(['match_id', 'innings'])['runs_this_over'].cumsum()
over_stats['cumulative_wickets'] = over_stats.groupby(['match_id', 'innings'])['wickets_this_over'].cumsum()

# Runs in last 5 overs (rolling window)
over_stats['runs_last5'] = (
    over_stats.groupby(['match_id', 'innings'])['runs_this_over']
    .transform(lambda x: x.rolling(5, min_periods=1).sum())
)

# Final score of the innings = target variable
final_scores = over_stats.groupby(['match_id', 'innings'])['cumulative_runs'].transform('max')
over_stats['final_score'] = final_scores

# Only train on data from over 5 onwards (too early = not enough info)
over_stats = over_stats[over_stats['over_num'] >= 5].copy()

print(f"Training samples: {len(over_stats)}")
print(over_stats[['batting_team', 'over_num', 'cumulative_runs', 'cumulative_wickets', 'final_score']].head(5))


# ─────────────────────────────────────────
# STEP 3: Encode Categorical Columns
# ML models need numbers, not text
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Encoding Categorical Columns")
print("=" * 50)

le_bat = LabelEncoder()
le_bowl = LabelEncoder()
le_venue = LabelEncoder()

over_stats['batting_team_enc'] = le_bat.fit_transform(over_stats['batting_team'])
over_stats['bowling_team_enc'] = le_bowl.fit_transform(over_stats['bowling_team'])
over_stats['venue_enc'] = le_venue.fit_transform(over_stats['venue'])

print("Teams encoded:", dict(zip(le_bat.classes_, le_bat.transform(le_bat.classes_))))


# ─────────────────────────────────────────
# STEP 4: Define Features & Target
# ─────────────────────────────────────────
FEATURES = [
    'batting_team_enc',   # which team is batting
    'bowling_team_enc',   # which team is bowling
    'venue_enc',          # where the match is
    'over_num',           # current over
    'cumulative_runs',    # runs scored so far
    'cumulative_wickets', # wickets lost so far
    'runs_last5',         # runs in last 5 overs (momentum)
]
TARGET = 'final_score'

X = over_stats[FEATURES]
y = over_stats[TARGET]


# ─────────────────────────────────────────
# STEP 5: Train-Test Split
# 80% for training, 20% for testing
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining rows: {len(X_train)}, Test rows: {len(X_test)}")


# ─────────────────────────────────────────
# STEP 6: Model 1 — Linear Regression
# Simplest model: draws a straight line
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: Linear Regression")
print("=" * 50)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

print(f"MAE  (avg error in runs): {lr_mae:.2f}")
print(f"R²   (how well it fits):  {lr_r2:.4f}")


# ─────────────────────────────────────────
# STEP 7: Model 2 — Random Forest
# Better: uses many decision trees together
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 7: Random Forest")
print("=" * 50)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print(f"MAE  (avg error in runs): {rf_mae:.2f}")
print(f"R²   (how well it fits):  {rf_r2:.4f}")


# ─────────────────────────────────────────
# STEP 8: Compare Models (Bar Chart)
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("IPL Score Predictor — Model Comparison", fontsize=14, fontweight='bold')

models = ['Linear Regression', 'Random Forest']
maes = [lr_mae, rf_mae]
r2s = [lr_r2, rf_r2]
colors = ['#4C72B0', '#55A868']

axes[0].bar(models, maes, color=colors)
axes[0].set_title("Mean Absolute Error (lower = better)")
axes[0].set_ylabel("MAE (runs)")
for i, v in enumerate(maes):
    axes[0].text(i, v + 0.3, f"{v:.1f}", ha='center', fontweight='bold')

axes[1].bar(models, r2s, color=colors)
axes[1].set_title("R² Score (higher = better, max=1.0)")
axes[1].set_ylabel("R²")
axes[1].set_ylim(0, 1.1)
for i, v in enumerate(r2s):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120, bbox_inches='tight')
print("\nChart saved: model_comparison.png")


# ─────────────────────────────────────────
# STEP 9: Feature Importance (Random Forest)
# Which features matter most?
# ─────────────────────────────────────────
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True)

fig2, ax = plt.subplots(figsize=(8, 4))
ax.barh(feat_df['Feature'], feat_df['Importance'], color='#55A868')
ax.set_title("Feature Importance (Random Forest)", fontweight='bold')
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120, bbox_inches='tight')
print("Chart saved: feature_importance.png")


# ─────────────────────────────────────────
# STEP 10: Make a Prediction
# Try predicting a live match scenario
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 10: Predict a Score")
print("=" * 50)

# Example: MI is batting vs CSK at Wankhede, over 10
# They've scored 85 runs, lost 2 wickets, 35 in last 5 overs
sample = {
    'batting_team': 'MI',
    'bowling_team': 'GT',
    'venue': 'Narendra Modi Stadium, Ahmedabad',
    'over_num': 15,
    'cumulative_runs': 122,
    'cumulative_wickets': 4,
    'runs_last5': 47,
}

def predict_score(sample_dict, model, le_bat, le_bowl, le_venue, feature_cols):
    # Encode the teams and venue
    try:
        bat_enc = le_bat.transform([sample_dict['batting_team']])[0]
        bowl_enc = le_bowl.transform([sample_dict['bowling_team']])[0]
        venue_enc = le_venue.transform([sample_dict['venue']])[0]
    except ValueError as e:
        print(f"  ⚠  Unknown label: {e}")
        return None

    row = [[
        bat_enc, bowl_enc, venue_enc,
        sample_dict['over_num'],
        sample_dict['cumulative_runs'],
        sample_dict['cumulative_wickets'],
        sample_dict['runs_last5'],
    ]]
    return model.predict(row)[0]

print(f"  Batting team : {sample['batting_team']}")
print(f"  Bowling team : {sample['bowling_team']}")
print(f"  Venue        : {sample['venue']}")
print(f"  Over         : {sample['over_num']}")
print(f"  Runs so far  : {sample['cumulative_runs']}")
print(f"  Wickets lost : {sample['cumulative_wickets']}")
print(f"  Last 5 overs : {sample['runs_last5']} runs")
print()

lr_score = predict_score(sample, lr, le_bat, le_bowl, le_venue, FEATURES)
rf_score = predict_score(sample, rf, le_bat, le_bowl, le_venue, FEATURES)

if lr_score: print(f"  Linear Regression predicts: {lr_score:.0f} runs")
if rf_score: print(f"  Random Forest predicts    : {rf_score:.0f} runs")

print("\nDone!")