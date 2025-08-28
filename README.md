# ⚽ Serie A Match Predictor

A machine learning project that predicts the outcomes of Serie A matches using historical performance data, team trends, and contextual features. The model is trained on past seasons and tested on recent fixtures to provide **data-driven insights** into win, draw, and loss probabilities.

---

## 📊 How It Works
- **Data preprocessing**: Converts match data (venue, opponent, referee, formation) into numerical codes.  
- **Rolling averages**: Computes 5-match rolling averages for key stats (goals, shots, xG, penalties, distance covered, etc.) to capture recent form.  
- **Feature set**: Includes contextual info (venue, day of week, kickoff hour) plus rolling stats.  
- **Model**: Random Forest Classifier (`n_estimators=50`, `min_samples_split=10`) trained with a chronological split to avoid data leakage.  

---

## ✅ Results
The model was tested on fixtures after **June 1, 2024**, giving the following metrics:

- **Accuracy**: ~47.8%  
- **Precision (weighted)**: ~46.7%  
### Confusion Matrix
| Actual \\ Predicted | 0 (Loss) | 1 (Draw) | 3 (Win) |
|---------------------|----------|----------|---------|
| **0 (Loss)**        | 161      | 37       | 72      |
| **1 (Draw)**        | 109      | 37       | 68      |
| **3 (Win)**         | 90       | 18       | 163     |

A confusion matrix shows where predictions succeed and fail, with the model strongest at identifying **wins and losses**, and less accurate on **draws** (a notoriously hard outcome to predict).  

---

## 🔎 What These Results Mean
- Random guessing across 3 outcomes would yield ~33% accuracy.  
- This predictor outperforms this baseline, reaching ~47.8%.  
- Draws remain underpredicted, a common challenge in football models due to their lower frequency.  

---

## ⚔️ Comparison to Other Models
- **Simple statistical baselines** (e.g., always predicting the home team or the most common outcome) achieve ~35–38% accuracy.  
- **The Random Forest model** surpasses these with ~47.8%.  
- **More advanced models** (e.g., gradient boosting or neural networks with betting market odds) can sometimes reach **50–55%** in research contexts, but at the cost of interpretability.  

---

## 🚀 Next Steps
- Experiment with **LightGBM** for boosted performance.  
- Incorporate **betting odds** and **player-level data**.  
- Extend to other leagues for broader benchmarking.  

---

## 🛠️ Usage
```bash
# Clone repo
git clone https://github.com/yourusername/seriea-predictor.git
cd seriea-predictor

# Install dependencies
pip install -r requirements.txt

# Run predictor
python SerieAPredictor.py

