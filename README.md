<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Outfit&weight=700&size=38&pause=1000&color=F5A623&center=true&vCenter=true&width=600&lines=🏏+IPL+Score+Predictor;Predict+Scores+with+ML;IPL+2026+Season+Data" alt="Typing SVG" />

<br/>

**A machine learning system that predicts IPL T20 innings totals in real-time,**  
**using live match context — overs bowled, runs scored, wickets fallen, and recent momentum.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![IPL 2026](https://img.shields.io/badge/IPL-2026%20Season-F5A623?style=for-the-badge)](https://ipl.com)

<br/>

```
🏆 Random Forest  →  MAE: 9.31 runs  |  R²: 0.865  |  Trained on 50 matches, 11,683 deliveries
```

</div>

---

## ✨ Key Highlights

<table>
<tr>
<td align="center" width="25%">
<h3>📦 Dataset</h3>
<b>11,683</b> deliveries<br/>
<b>50</b> IPL 2026 matches<br/>
<b>10</b> teams · <b>11</b> venues
</td>
<td align="center" width="25%">
<h3>🎯 Best Model</h3>
Random Forest<br/>
<b>MAE: ±9.3 runs</b><br/>
<b>R²: 0.865</b>
</td>
<td align="center" width="25%">
<h3>⚡ Features</h3>
<b>7</b> engineered features<br/>
Real-time match state<br/>
Rolling 5-over momentum
</td>
<td align="center" width="25%">
<h3>🔬 Models</h3>
Linear Regression<br/>
Random Forest<br/>
Side-by-side benchmarking
</td>
</tr>
</table>

---

## 📊 Results at a Glance

<div align="center">

| | Model | Mean Absolute Error ↓ | R² Score ↑ | Verdict |
|---|---|---|---|---|
| 🔵 | Linear Regression | 16.03 runs | 0.7041 | Good baseline |
| 🌲 | **Random Forest** | **9.31 runs** | **0.8650** | ✅ **Best model** |

> **Random Forest is 42% more accurate on MAE** and explains **86.5% of score variance**

</div>

<br/>

| Model Comparison | Feature Importance |
|:---:|:---:|
| ![Model Comparison](model_comparison.png) | ![Feature Importance](feature_importance.png) |
| *MAE & R² across both models* | *Which features drive predictions most* |

---

## 🎯 Live Prediction Example

> **Scenario:** MI batting vs GT at Narendra Modi Stadium, Ahmedabad — Over 15, 122/4, 47 runs in last 5 overs

```python
# Input match state
sample = {
    'batting_team':       'MI',
    'bowling_team':       'GT',
    'venue':              'Narendra Modi Stadium, Ahmedabad',
    'over_num':           15,
    'cumulative_runs':    122,
    'cumulative_wickets': 4,
    'runs_last5':         47
}

# Predicted final scores
# ├── 🔵 Linear Regression  →  166 runs
# └── 🌲 Random Forest      →  194 runs  ✅
```

---

## 🔬 ML Pipeline

```mermaid
flowchart TD
    A[📄 ipl_2026_deliveries.csv\n11683 deliveries · 50 matches] --> B

    B[🧹 Step 1 · Load & Inspect\nVerify teams, shape, nulls] --> C

    C[⚙️ Step 2 · Feature Engineering\nGroup by match+innings+over\nCompute cumulative runs, wickets\nRolling 5-over momentum window] --> D

    D[🔢 Step 3 · Label Encoding\nTeams & venues → integers\nvia scikit-learn LabelEncoder] --> E

    E[🎯 Step 4 · Define Features & Target\n7 features → final_score\nFilter: over ≥ 5 only] --> F

    F[✂️ Step 5 · Train/Test Split\n80% train · 1102 rows\n20% test  · 276 rows] --> G & H

    G[🔵 Step 6 · Linear Regression\nMAE: 16.03 · R²: 0.7041] --> I
    H[🌲 Step 7 · Random Forest\n100 trees · MAE: 9.31 · R²: 0.865] --> I

    I[📊 Step 8 & 9 · Visualise\nmodel_comparison.png\nfeature_importance.png] --> J

    J[🏏 Step 10 · Live Prediction\nFeed match state → get score]

    style A fill:#1e3a5f,color:#fff
    style J fill:#14532d,color:#fff
    style H fill:#166534,color:#fff
    style G fill:#1e40af,color:#fff
```

---

## 🧠 Features Explained

| # | Feature | Type | What It Captures |
|---|---------|------|-----------------|
| 1 | `batting_team_enc` | Categorical | Team batting strength & style |
| 2 | `bowling_team_enc` | Categorical | Bowling attack quality |
| 3 | `venue_enc` | Categorical | Ground dimensions & pitch characteristics |
| 4 | `over_num` | Numeric | Innings stage — powerplay vs. death overs |
| 5 | `cumulative_runs` | Numeric | **#1 predictor** — score trajectory so far |
| 6 | `cumulative_wickets` | Numeric | Batting resources remaining |
| 7 | `runs_last5` | Numeric | Scoring momentum of last 5 overs |

> 🎯 **Target:** `final_score` — the innings total. Predictions only made from **over 5 onwards** to ensure enough context.

---

## 📂 Project Structure

```
score-predictor/
│
├── 🐍  model.py                   ← Full 10-step ML pipeline
├── 📊  ipl_2026_deliveries.csv    ← Ball-by-ball dataset (11,683 rows)
├── 📈  model_comparison.png       ← MAE & R² bar chart
├── 🌿  feature_importance.png     ← Random Forest feature rankings
├── 📋  output.txt                 ← Latest console run output
└── 📖  README.md                  ← You are here
```

---

## 📦 Dataset Details

| Property | Value |
|----------|-------|
| **Season** | IPL 2026 |
| **Phase** | Group Stage |
| **Date Range** | Mar 28 – May 7, 2026 |
| **Total Deliveries** | 11,683 |
| **Total Matches** | 50 |
| **Teams** | CSK · DC · GT · KKR · LSG · MI · PBKS · RCB · RR · SRH |
| **Venues** | 11 stadiums across India |
| **Training Samples** | 1,378 over-level snapshots |
| **Train / Test Split** | 1,102 / 276 rows (80/20) |

<details>
<summary>🏟️ <b>View all 11 venues</b></summary>

<br/>

| Stadium | City |
|---------|------|
| M.Chinnaswamy Stadium | Bengaluru |
| MA Chidambaram Stadium | Chennai |
| Arun Jaitley Stadium | Delhi |
| Eden Gardens | Kolkata |
| Wankhede Stadium | Mumbai |
| Narendra Modi Stadium | Ahmedabad |
| Rajiv Gandhi International Stadium | Hyderabad |
| Sawai Mansingh Stadium | Jaipur |
| Barsapara Cricket Stadium | Guwahati |
| Ekana Cricket Stadium | Lucknow |
| Maharaja Yadavindra Singh International Cricket Stadium | Mullanpur |

</details>

---

## 🚀 Getting Started

### 1 · Clone the repo

```bash
git clone https://github.com/thunder-11/score-predictor.git
cd score-predictor
```

### 2 · Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 3 · Run the full pipeline

```bash
python model.py
```

<details>
<summary>📋 <b>What happens when you run it?</b></summary>

```
✅  Loads 11,683 ball-by-ball deliveries from CSV
✅  Engineers per-over features (cumulative stats + momentum)
✅  Label-encodes teams and venues
✅  Trains Linear Regression  →  MAE: 16.03 · R²: 0.7041
✅  Trains Random Forest      →  MAE:  9.31 · R²: 0.8650
✅  Saves model_comparison.png and feature_importance.png
✅  Runs a live score prediction for a sample match state
```

</details>

---

## 🔧 Predict Any Match State

Simply edit the `sample` dict near the bottom of `model.py`:

```python
sample = {
    'batting_team':       'RCB',                           # any of the 10 teams
    'bowling_team':       'CSK',
    'venue':              'M.Chinnaswamy Stadium, Bengaluru',
    'over_num':           12,                              # overs 5–20
    'cumulative_runs':    98,                              # runs so far
    'cumulative_wickets': 3,                               # wickets fallen
    'runs_last5':         42,                              # last 5 overs runs
}
```

---

## 🌲 Why Random Forest Outperforms Linear Regression

Cricket is **inherently non-linear**:

- A cluster of wickets mid-innings can collapse a score by 30+ runs
- Death-over hitting (overs 17–20) can add 60+ runs in a blink
- Venue and team matchup effects are non-additive

**Linear Regression** assumes all relationships are straight lines — a major handicap for cricket data.

**Random Forest** grows 100 independent decision trees, each learning different non-linear split patterns, then averages their predictions. This ensemble approach naturally handles the spiky, context-dependent nature of T20 scoring.

| Metric | Linear Regression | Random Forest | Δ Improvement |
|--------|:-----------------:|:-------------:|:---:|
| MAE (runs) | 16.03 | **9.31** | **−42%** ✅ |
| R² Score | 0.7041 | **0.8650** | **+23%** ✅ |

---

## 🛠️ Tech Stack

| Library | Version | Role |
|---------|---------|------|
| **Python** | 3.8+ | Core language |
| **Pandas** | 2.x | Data wrangling, groupby, rolling windows |
| **NumPy** | 1.x | Array operations |
| **scikit-learn** | 1.x | Models, LabelEncoder, train_test_split, metrics |
| **Matplotlib** | 3.x | Bar charts, feature importance plots |

---

## 🔮 Roadmap

- [ ] 🚀 **XGBoost / LightGBM** — gradient boosting for even higher accuracy
- [ ] 🧬 **Player-level features** — striker strike rate, bowler economy in this innings
- [ ] 🌦️ **Contextual features** — toss result, day/night, pitch report
- [ ] 🌐 **Streamlit web app** — interactive real-time score predictor
- [ ] 📡 **REST API** — plug into live match dashboards
- [ ] 📅 **Multi-season data** — extend to IPL 2020–2026 for robustness
- [ ] 🏆 **Playoffs coverage** — include Qualifier & Final matches

---

## 🤝 Contributing

All contributions are welcome — from bug fixes to new model ideas!

```bash
# 1. Fork this repository
# 2. Create your feature branch
git checkout -b feature/add-xgboost

# 3. Make your changes and commit
git commit -m "feat: add XGBoost regressor with hyperparameter tuning"

# 4. Push and open a Pull Request
git push origin feature/add-xgboost
```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

<br/>

**Built with ❤️, Python, and a deep love for cricket**

<br/>

*Found this useful? Drop a ⭐ — it keeps the stumps standing!*

<br/>

[![GitHub stars](https://img.shields.io/github/stars/thunder-11/score-predictor?style=social)](https://github.com/thunder-11/score-predictor)

</div>
