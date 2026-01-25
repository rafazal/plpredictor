## Premier League Match Predictor

A Python-based project that predicts **Premier League match scorelines** using historical data and team form.

The model processes multiple seasons of match data, engineers features like **recent form and goal differential**, and trains separate models to predict **home and away goals**, allowing it to output full scoreline predictions rather than just match results.

### Tech Stack

* Python
* pandas
* scikit-learn
* NumPy
* RandomForestRegressor

### How It Works

1. Match data is cleaned and organized by team and matchweek
2. Recent performance features are generated using rolling statistics
3. Separate models predict home and away goals for each fixture
4. Predictions can be surfaced by matchweek as the season progresses

Built as a practical sports analytics project, with room to expand into a full-stack web app.

