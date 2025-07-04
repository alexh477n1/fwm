import pandas as pd
from textblob import TextBlob


def analyze_comments(path="data/player_comments.csv"):
    df = pd.read_csv(path)
    df["sentiment"] = df["comment"].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df


def main():
    df = analyze_comments()
    df.to_csv("data/player_sentiment.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
