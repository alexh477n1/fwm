# Football Player Market Value Estimation

This project demonstrates a simple workflow for estimating football player market values using publicly available data.

## Dataset

Sample data files are stored in the `data/` folder. The training set `rf_train.csv` was sourced from [this GitHub repository](https://github.com/vgvr0/Market_value_football_players_24). It contains basic player metrics and their market values in millions of euros.

A predictions file is produced after running the model.

## Usage

1. Install requirements:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```
2. Run the valuation script:
   ```bash
   python valuation.py
   ```
   Metrics will be printed and predictions saved to `data/predictions.csv`.

This setup provides a starting point for building a more advanced player valuation pipeline.
