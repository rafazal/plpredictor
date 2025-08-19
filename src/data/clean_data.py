import pandas as pd
import os

def load_and_clean_data(path="data/raw/fbref_2024.csv"):
    # Load CSV
    df = pd.read_csv(path)

    # Keep only rows where a match result exists
    df = df[df["FTR"].notna()].copy()

    # Rename and standardize columns
    df.rename(columns={
        "HomeTeam": "HomeTeam",
        "AwayTeam": "AwayTeam",
        "FTHG": "HomeGoals",
        "FTAG": "AwayGoals",
        "FTR": "Result",         # H/D/A
        "Date": "Date"
    }, inplace=True)

    # Convert date format if needed
    try:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    except:
        df["Date"] = pd.to_datetime(df["Date"])  # fallback

    # Optional: Keep only relevant columns
    columns_to_keep = [
        "Date", "HomeTeam", "AwayTeam",
        "HomeGoals", "AwayGoals", "Result"
    ]
    df = df[columns_to_keep]

    # Save cleaned version
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_matches.csv", index=False)
    print("✅ Cleaned dataset saved to data/processed/cleaned_matches.csv")

    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())
