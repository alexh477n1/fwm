import pandas as pd
from pathlib import Path
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor

TRAIN_PATH = Path("data/rf_train.csv")
CHAMP_PATH = Path("data/championship_players.csv")
COMMENTS_PATH = Path("data/player_comments.csv")


def train_model(path=TRAIN_PATH):
    df = pd.read_csv(path)
    X = df.drop("value_eur_m", axis=1)
    y = df["value_eur_m"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model


def predict_for_championship(model, path=CHAMP_PATH):
    players = pd.read_csv(path)
    X = players.drop("name", axis=1)
    players["predicted_value"] = model.predict(X)
    players["future_star_index"] = (
        (players["potential"] - players["overall"]) * players["predicted_value"]
        / players["age"]
    )
    return players


def add_sentiment(players, path=COMMENTS_PATH):
    comments = pd.read_csv(path)
    comments["sentiment"] = comments["comment"].apply(lambda c: TextBlob(c).sentiment.polarity)
    merged = players.merge(comments[["player", "sentiment"]], left_on="name", right_on="player", how="left")
    merged.drop(columns=["player"], inplace=True)
    merged["sentiment"].fillna(0, inplace=True)
    merged["star_score"] = merged["future_star_index"] * (1 + merged["sentiment"])
    return merged


def main():
    model = train_model()
    players = predict_for_championship(model)
    final = add_sentiment(players)
    final.to_csv("data/championship_predictions.csv", index=False)
    print(final.head())


if __name__ == "__main__":
    main()
