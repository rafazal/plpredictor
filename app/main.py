from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson
from collections import Counter
import datetime
from zoneinfo import ZoneInfo
import datetime as dt

TZ = ZoneInfo("America/Chicago")
MATCHWEEK_RELEASE_HOUR_LOCAL = 0  # show predictions at 09:00 local on the week start date

app = Flask(__name__)

# === Load models and data ===
home_model = joblib.load('models/home_model.pkl')
away_model = joblib.load('models/away_model.pkl')

with open('models/team_to_id.pkl', 'rb') as f:
    team_to_id = joblib.load(f)

with open('models/team_stats_dict.pkl', 'rb') as f:
    team_stats_dict = joblib.load(f)

fixtures_df = pd.read_csv("/Users/rafay/Documents/pl-predictor/data/epl-2025-GMTStandardTime.csv")
fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'], errors='coerce')
today = pd.to_datetime(datetime.date.today())

# === Helper Functions ===
def get_stat(stats, key, default=1.5):
    return stats.get(key, default)

def simulate_poisson_result(home_xg, away_xg, n_simulations=200):
    outcomes = []
    for _ in range(n_simulations):
        hg = poisson.rvs(mu=home_xg)
        ag = poisson.rvs(mu=away_xg)
        hg = np.clip(hg, 0, 5)
        ag = np.clip(ag, 0, 5)
        outcomes.append((hg, ag))
    most_common = Counter(outcomes).most_common(1)[0][0]
    return most_common

def predict_score(home_team, away_team):
    home_id = team_to_id.get(home_team)
    away_id = team_to_id.get(away_team)

    home_stats = team_stats_dict.get(home_team, {})
    away_stats = team_stats_dict.get(away_team, {})

    X_input = pd.DataFrame([{
        'HomeTeamID': home_id,
        'AwayTeamID': away_id,
        'HomeRollingGF': get_stat(home_stats, 'AvgGF'),
        'HomeRollingGA': get_stat(home_stats, 'AvgGA'),
        'HomeRollingPoints': get_stat(home_stats, 'PointsPerMatch'),
        'HomeFormScore': get_stat(home_stats, 'WinRate') * 10,
        'HomeGDForm': get_stat(home_stats, 'GoalDifference'),
        'AwayRollingGF': get_stat(away_stats, 'AvgGF'),
        'AwayRollingGA': get_stat(away_stats, 'AvgGA'),
        'AwayRollingPoints': get_stat(away_stats, 'PointsPerMatch'),
        'AwayFormScore': get_stat(away_stats, 'WinRate') * 10,
        'AwayGDForm': get_stat(away_stats, 'GoalDifference'),
    }])

    home_xg = home_model.predict(X_input)[0]
    away_xg = away_model.predict(X_input)[0]

    home_goals, away_goals = simulate_poisson_result(home_xg, away_xg)
    return int(home_goals), int(away_goals)

def week_start_datetime_local(week_num: int):
    """Return the local datetime when predictions for a matchweek should unlock.
       Uses the earliest fixture date for the week, at MATCHWEEK_RELEASE_HOUR_LOCAL."""
    week_df = fixtures_df[fixtures_df['Round Number'] == week_num].copy()
    if week_df.empty:
        return None

    # Parse dates and find the first fixture date
    dates = pd.to_datetime(week_df['Date'], errors='coerce')
    first = dates.min()
    if pd.isna(first):
        return None

    # Treat CSV date as a local calendar date (no timezone). Unlock at chosen hour that day.
    local_date = first.date()
    unlock_dt = dt.datetime.combine(local_date, dt.time(hour=MATCHWEEK_RELEASE_HOUR_LOCAL, minute=0))
    return unlock_dt.replace(tzinfo=TZ)

def should_release_week(week_num: int) -> bool:
    """True if we’re past the unlock time for the given matchweek."""
    unlock = week_start_datetime_local(week_num)
    if unlock is None:
        return False
    now_local = dt.datetime.now(TZ)
    return now_local >= unlock

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html', team_names=sorted(team_to_id.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']

    if home_team == away_team:
        return render_template('result.html', prediction="Invalid: Same team selected.")

    home_goals, away_goals = predict_score(home_team, away_team)
    prediction = f"{home_team} {home_goals} - {away_goals} {away_team}"
    return render_template('result.html', prediction=prediction)

@app.route('/matchweek/<int:week_num>')
@app.route('/matchweek/<int:week_num>')
def show_matchweek(week_num):
    week_fixtures = fixtures_df[fixtures_df['Round Number'] == week_num]
    matches = []

    # Decide once per page whether predictions are visible
    release_now = should_release_week(week_num)

    for _, row in week_fixtures.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        date = pd.to_datetime(row['Date'], errors='coerce')

        if not release_now:
            prediction = "Upcoming"
        else:
            try:
                home_goals, away_goals = predict_score(home_team, away_team)
                prediction = f"{home_goals} - {away_goals}"
            except Exception as e:
                print(f"Prediction error for {home_team} vs {away_team}: {e}")
                prediction = "Upcoming"

        matches.append({
            'date': date.strftime("%Y-%m-%d") if pd.notna(date) else "TBD",
            'home': home_team,
            'away': away_team,
            'prediction': prediction
        })

    return render_template('matchweek.html', matchweek=week_num, matches=matches)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)