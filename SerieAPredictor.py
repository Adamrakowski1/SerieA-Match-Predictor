import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
"""
The rolling_averags function averages performance metrics for each team
The function takes each team, sorts data by date, and computes 5 match
rolling averages for each team using specified columns
This allows us to base predictions on how a team is performing recently
"""
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(5, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# Load and process data from csv
matches = pd.read_csv("matches_serie_A.csv", index_col=0)
matches["Date"] = pd.to_datetime(matches["Date"], format="%Y-%m-%d", errors="coerce") 

# Create features that will be used by the machine learning algorithms
# Convert these objects to numerical values
matches["venue_code"] = matches["Venue"].astype("category").cat.codes
matches["opponent_codes"] = matches["Opponent"].astype("category").cat.codes
matches["formation_code"] = matches["Formation"].astype("category").cat.codes
matches["opp_formation_code"] = matches["Opp Formation"].astype("category").cat.codes
matches["referee_code"] = matches["Referee"].astype("category").cat.codes

# Get the hour from the overall time and assign the day to an integer
matches["hour"] = (matches["Time"].str.split(":").str[0]).astype(int)
matches["day_code"] = matches["Date"].dt.dayofweek

# Assign points for each result
matches["target"] = matches["Result"].map({"W": 3, "D": 1, "L": 0}).astype(int)

# Data that will go into rolling averages
cols = ["GF","GA","xG","Sh","SoT","Dist","FK","PK","PKatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Create rolling averages for each team separately
matches_rolling = pd.DataFrame()

# Loop through each team and calculate rolling averages
for team in matches["Team"].unique():
    team_data = matches[matches["Team"] == team].copy()
    team_rolling = rolling_averages(team_data, cols, new_cols)
    matches_rolling = pd.concat([matches_rolling, team_rolling], ignore_index=True)

# Sort by date for easy reading
matches_rolling = matches_rolling.sort_values("Date").reset_index(drop=True)

"""
Split the data into training and testing sets using a chronological split
This makes sure we don't have a data leak
"""
train = matches_rolling[matches_rolling["Date"] < "2024-06-01"]
test = matches_rolling[matches_rolling["Date"] >= "2024-06-01"]


# Define safe predictors (no info about current match result, no data leakage)
safe_predictors = [
    "venue_code", "opponent_codes", "hour", "day_code",
    "formation_code", "opp_formation_code",
    "GF_rolling", "GA_rolling", "Sh_rolling", "SoT_rolling", 
    "Dist_rolling", "FK_rolling", "PK_rolling", "PKatt_rolling",
    "referee_code"
]


"""
Train the Random Forest model and evaluate its performance on the test set
This is a baseline performance to use for comparison
"""
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(train[safe_predictors], train["target"])
# Make predictions
preds = rf.predict(test[safe_predictors])

# Results
accuracy = accuracy_score(test["target"], preds)
precision = precision_score(test["target"], preds, average='weighted')
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")

"""
Matrix to show detailed predictions
"""
combined = pd.DataFrame({"actual": test["target"], "predicted": preds})
print(pd.crosstab(index=combined["actual"], columns=combined["predicted"]))