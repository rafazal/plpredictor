import pandas as pd
import joblib
import os

# Load the cleaned dataset (adjusted path)
df = pd.read_csv("data/processed/cleaned_matches.csv")

# Create dictionary of average team-level stats
team_stats_dict = {}

teams = set(df["HomeTeam"]).union(set(df["AwayTeam"]))

for team in teams:
    team_home = df[df["HomeTeam"] == team]
    team_away = df[df["AwayTeam"] == team]

    avg_GF = pd.concat([team_home["HomeGoals"], team_away["AwayGoals"]]).mean()
    avg_GA = pd.concat([team_home["AwayGoals"], team_away["HomeGoals"]]).mean()

    team_stats_dict[team] = {
        "AvgGF": round(avg_GF, 2),
        "AvgGA": round(avg_GA, 2)
    }

# Save to a pickle file
os.makedirs("models", exist_ok=True)
joblib.dump(team_stats_dict, "models/team_stats_dict.pkl")

print("✅ team_stats_dict.pkl created successfully!")
