<div align="center">

# 🏏 IPL Score Predictor

### Predict IPL innings scores in real-time using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![IPL 2026](https://img.shields.io/badge/IPL-2026%20Season-gold?style=for-the-badge)](https://ipl.com)

> **Given the current match state — over number, runs scored, wickets lost, recent momentum — this model predicts the final innings score with high accuracy.**

</div>

---

## 📸 Results at a Glance

<div align="center">

| Model | MAE ↓ | R² Score ↑ |
|---|---|---|
| 🔵 Linear Regression | 16.03 runs | 0.7041 |
| 🌲 **Random Forest** | **9.31 runs** | **0.8650** |

*Random Forest outperforms Linear Regression by **42% on MAE** and **23% on R²***

</div>

| Model Comparison | Feature Importance |
|---|---|
| ![Model Comparison](model_comparison.png) | ![Feature Importance](feature_importance.png) |

---

## 🎯 Live Prediction Demo

```
📍 Scenario: MI vs GT at Narendra Modi Stadium, Ahmedabad
⚡ Over 15 | 122/4 | 47 runs in last 5 overs

  🔵 Linear Regression  →  166 runs
  🌲 Random Forest      →  194 runs  ✅ (recommended)
```

---

## 📂 Project Structure

```
score-predictor/
│
├── 📄 model.py                    # Full ML pipeline (data → features → train → predict)
├── 📊 ipl_2026_deliveries.csv     # Ball-by-ball IPL 2026 dataset (11,683 deliveries, 50 matches)
├── 📈 model_comparison.png        # MAE & R² bar chart comparing both models
├── 🌿 feature_importance.png      # Which features matter most (Random Forest)
├── 📋 output.txt                  # Full console output from the last model run
└── 📖 README.md                   # You are here
```

---

## 🔬 How It Works

The pipeline follows **10 well-commented steps**:

```
Raw Deliveries CSV
       │
       ▼
① Load Data          →  11,683 deliveries across 50 IPL 2026 matches (Group Stage)
       │
       ▼
② Feature Engineering →  Aggregate ball-by-ball data into per-over snapshots
       │                  (cumulative runs, wickets, rolling 5-over momentum)
       ▼
③ Label Encoding      →  Teams & venues encoded to integers (ML-ready)
       │
       ▼
④ Define Features     →  7 meaningful predictors selected
       │
       ▼
⑤ Train/Test Split    →  80% train (1,102 rows), 20% test (276 rows)
       │
       ▼
⑥ Linear Regression   →  Baseline model — fast & interpretable
       │
       ▼
⑦ Random Forest       →  100 decision trees — powerful & accurate
       │
       ▼
⑧ Model Comparison    →  Side-by-side MAE & R² chart saved as PNG
       │
       ▼
⑨ Feature Importance  →  Which inputs drive the prediction most?
       │
       ▼
⑩ Live Prediction     →  Feed a match state → get predicted final score
```

---

## 🧠 Features Used

| # | Feature | Description |
|---|---------|-------------|
| 1 | `batting_team_enc` | Which team is batting (label-encoded) |
| 2 | `bowling_team_enc` | Which team is bowling (label-encoded) |
| 3 | `venue_enc` | Stadium / ground (label-encoded) |
| 4 | `over_num` | Current over number (5–20) |
| 5 | `cumulative_runs` | Total runs scored so far in the innings |
| 6 | `cumulative_wickets` | Wickets lost so far |
| 7 | `runs_last5` | Runs scored in the last 5 overs (momentum) |

> **Target variable:** `final_score` — the actual innings total at the end.  
> Training is restricted to overs **5+** to ensure sufficient context.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | IPL 2026 — ball-by-ball deliveries |
| **File** | `ipl_2026_deliveries.csv` |
| **Deliveries** | 11,683 |
| **Matches** | 50 |
| **Phase** | Group Stage (Mar 28 – May 7, 2026) |
| **Teams** | CSK · DC · GT · KKR · LSG · MI · PBKS · RCB · RR · SRH |
| **Venues** | 11 stadiums across India |
| **Training Samples** | 1,378 over-level snapshots |

### 🏟️ Venues Covered

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
| Bharat Ratna Sri Atal Bihari Vajpayee Ekana Stadium | Lucknow |
| Maharaja Yadavindra Singh International Cricket Stadium | Mullanpur |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/thunder-11/score-predictor.git
cd score-predictor
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 3. Run the Pipeline

```bash
python model.py
```

This will:
- ✅ Load and process the dataset
- ✅ Train both models
- ✅ Print performance metrics
- ✅ Save `model_comparison.png` and `feature_importance.png`
- ✅ Output a live prediction for a sample match scenario

---

## 🔧 Customize a Prediction

Edit the `sample` dictionary at the bottom of `model.py` to predict for any scenario:

```python
sample = {
    'batting_team': 'RCB',         # Any of the 10 IPL teams
    'bowling_team': 'CSK',
    'venue': 'M.Chinnaswamy Stadium, Bengaluru',
    'over_num': 12,                # Between 5 and 20
    'cumulative_runs': 98,         # Runs scored so far
    'cumulative_wickets': 3,       # Wickets fallen
    'runs_last5': 42,              # Runs in last 5 overs
}
```

---

## 🌲 Why Random Forest Wins

| Metric | Linear Regression | Random Forest | Improvement |
|--------|-------------------|---------------|-------------|
| MAE | 16.03 runs | **9.31 runs** | 42% better ✅ |
| R² | 0.7041 | **0.8650** | 23% better ✅ |

Cricket scoring is **non-linear** by nature — wickets cause sudden drops, death-over hitting causes spikes. Random Forest captures these complex patterns using an ensemble of 100 decision trees, while Linear Regression can only fit a straight line through the data.

---

## 📈 Feature Importance Insight

According to the Random Forest model, the most influential predictors (in order) are:

1. 🏆 **`cumulative_runs`** — How many runs have already been scored is the strongest signal
2. ⚡ **`over_num`** — The current over provides crucial innings-stage context
3. 🔥 **`runs_last5`** — Recent momentum is a strong predictor of final pace
4. 💀 **`cumulative_wickets`** — Wickets lost constrain the remaining batting depth
5. 🏟️ **`venue_enc`** — Some grounds are consistently higher/lower scoring
6. 🏏 **`batting_team_enc`** — Team identity captures batting strength
7. 🥎 **`bowling_team_enc`** — Bowling attack quality matters

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core language |
| **Pandas** | Data loading, groupby, rolling windows |
| **NumPy** | Numerical operations |
| **scikit-learn** | ML models, Label Encoding, train/test split, metrics |
| **Matplotlib** | Visualization — bar charts, horizontal feature importance |

---

## 🔮 Future Improvements

- [ ] Add **XGBoost / LightGBM** for potentially higher accuracy
- [ ] Incorporate **toss result**, **pitch conditions**, and **weather** features
- [ ] Build a **Streamlit web app** for interactive real-time predictions
- [ ] Extend to **multi-season dataset** (2020–2026) for better generalization
- [ ] Add **player-level features** (top batsman strike rate, bowler economy)
- [ ] Deploy as a **REST API** for live match integration
- [ ] Cover **Playoffs & Finals** phases once data is available

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/xgboost-model`)
3. Commit your changes (`git commit -m 'Add XGBoost model'`)
4. Push to the branch (`git push origin feature/xgboost-model`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">

**Made with ❤️ and cricket passion**

⭐ *If this project helped you, consider giving it a star!* ⭐

</div>
