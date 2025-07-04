# Football Player Market Value Estimation

This project demonstrates a simple workflow for estimating football player market values using publicly available data.  
The repository now includes utilities for gathering open match data, generating
visualisations and incorporating basic sentiment analysis when valuing players.

## Dataset

Sample data files are stored in the `data/` folder. The training set `rf_train.csv`
was sourced from [this GitHub repository](https://github.com/vgvr0/Market_value_football_players_24).
Additional mock data for Championship players and example comments are provided
for demonstration purposes.

A predictions file is produced after running the model.

## Usage

1. Install requirements:
   ```bash
   pip install pandas scikit-learn matplotlib textblob requests
   ```
2. Run the valuation script:
   ```bash
   python valuation.py
   ```
   Metrics will be printed and predictions saved to `data/predictions.csv`.
   A few plots are also written to the `data/` directory along with
   `data/top_future_stars.csv` which lists the players with the highest derived
   "future star" index.

3. Collect basic match counts for European leagues from the StatsBomb open data:
   ```bash
   python collect_statsbomb.py
   ```

4. Analyse Championship players and combine sentiment:
   ```bash
   python advanced_analysis.py
   ```
   This outputs `data/championship_predictions.csv` including a sentiment
   adjusted score.

This setup provides a starting point for building a more advanced player valuation pipeline.
