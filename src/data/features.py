import pandas as pd
import numpy as np

def add_rolling_stats(df, window=5):
    df = df.copy()
    
    # Sort matches by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    teams = df['HomeTeam'].unique()
    
    # Create blank DataFrames to store features
    home_features = []
    away_features = []

    for team in teams:
        # Filter matches involving the team
        team_home = df[df['HomeTeam'] == team].copy()
        team_away = df[df['AwayTeam'] == team].copy()

        # Calculate rolling stats
        team_home['RollingGF'] = team_home['HomeGoals'].rolling(window).mean().shift(1)
        team_home['RollingGA'] = team_home['AwayGoals'].rolling(window).mean().shift(1)
        team_home['RollingPoints'] = team_home.apply(lambda row: 3 if row['HomeGoals'] > row['AwayGoals'] else 1 if row['HomeGoals'] == row['AwayGoals'] else 0, axis=1)
        team_home['RollingPoints'] = team_home['RollingPoints'].rolling(window).sum().shift(1)

        team_away['RollingGF'] = team_away['AwayGoals'].rolling(window).mean().shift(1)
        team_away['RollingGA'] = team_away['HomeGoals'].rolling(window).mean().shift(1)
        team_away['RollingPoints'] = team_away.apply(lambda row: 3 if row['AwayGoals'] > row['HomeGoals'] else 1 if row['AwayGoals'] == row['HomeGoals'] else 0, axis=1)
        team_away['RollingPoints'] = team_away['RollingPoints'].rolling(window).sum().shift(1)

        # Assign to master list
        home_features.append(team_home)
        away_features.append(team_away)

    # Combine all rows back together
    df_full = pd.concat(home_features + away_features).sort_values('Date')

    # Rename columns for clarity
    df_full.rename(columns={
        'RollingGF': 'RollingGF',
        'RollingGA': 'RollingGA',
        'RollingPoints': 'RollingPoints'
    }, inplace=True)

    # Now we re-merge those values onto the main df
    df = df.merge(df_full[['Date', 'HomeTeam', 'RollingGF', 'RollingGA', 'RollingPoints']],
                  on=['Date', 'HomeTeam'], how='left').rename(columns={
                      'RollingGF': 'HomeRollingGF',
                      'RollingGA': 'HomeRollingGA',
                      'RollingPoints': 'HomeRollingPoints'
                  })

    df = df.merge(df_full[['Date', 'AwayTeam', 'RollingGF', 'RollingGA', 'RollingPoints']],
                  on=['Date', 'AwayTeam'], how='left').rename(columns={
                      'RollingGF': 'AwayRollingGF',
                      'RollingGA': 'AwayRollingGA',
                      'RollingPoints': 'AwayRollingPoints'
                  })

    # Home/Away Form Score (Points per match from last 5)
    df['HomeFormScore'] = df['HomeRollingPoints'] / window
    df['AwayFormScore'] = df['AwayRollingPoints'] / window

    # Form goal difference
    df['HomeGDForm'] = df['HomeRollingGF'] - df['HomeRollingGA']
    df['AwayGDForm'] = df['AwayRollingGF'] - df['AwayRollingGA']

    return df