import pandas as pd
import joblib

# 📥 Load your cleaned data
df = pd.read_csv("data/processed/cleaned_matches.csv")

# ✅ Create the team_stats_dict
teams = sorted(df['HomeTeam'].unique())
team_stats_dict = {}

for team in teams:
    team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    recent_matches = team_matches.sort_values('Date', ascending=False).head(5)

    gf = 0
    ga = 0
    pts = 0
    gd = 0
    for _, row in recent_matches.iterrows():
        if row['HomeTeam'] == team:
            gf += row['FTHG']
            ga += row['FTAG']
            if row['FTR'] == 'H':
                pts += 3
            elif row['FTR'] == 'D':
                pts += 1
            gd += row['FTHG'] - row['FTAG']
        else:
            gf += row['FTAG']
            ga += row['FTHG']
            if row['FTR'] == 'A':
                pts += 3
            elif row['FTR'] == 'D':
                pts += 1
            gd += row['FTAG'] - row['FTHG']
    
    team_stats_dict[team] = {
        'GF': gf / 5,
        'GA': ga / 5,
        'Points': pts / 5,
        'GD': gd / 5
    }

# 💾 Save the dictionary
joblib.dump(team_stats_dict, "models/team_stats_dict.pkl")

print("✅ team_stats_dict.pkl exported successfully!")
