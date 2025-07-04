import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


def load_data(path):
    return pd.read_csv(path)


def train_models(df):
    X = df.drop('value_eur_m', axis=1)
    y = df['value_eur_m']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_pred = lin_reg.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred)
    lin_mae = mean_absolute_error(y_test, lin_pred)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)

    metrics = {
        'linear_r2': lin_r2,
        'linear_mae': lin_mae,
        'rf_r2': rf_r2,
        'rf_mae': rf_mae,
    }

    predictions = pd.DataFrame({
        'actual': y_test,
        'linear_pred': lin_pred,
        'rf_pred': rf_pred
    })

    return metrics, predictions


def main():
    df = load_data('data/rf_train.csv')
    metrics, predictions = train_models(df)
    print('Metrics:', metrics)
    predictions.to_csv('data/predictions.csv', index=False)


if __name__ == '__main__':
    main()
