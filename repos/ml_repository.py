import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.statespace.tools import diff


def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col="date", parse_dates=True, sep=";")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.asfreq('B')
    df['value'] = df['value'].fillna(method='ffill')
    return df

def train_arima_with_params(file_path: str, save_path: str, order):
    df = load_dataset(file_path)
    model = SARIMAX(df["value"], order=order, freq='B')
    model_fit = model.fit(disp=False)
    with open(save_path, 'wb') as f:
        pickle.dump(model_fit, f)
    return save_path

def train_svr_with_params(file_path: str, save_path: str, kernel: str, C: float, epsilon: float):
    df = load_dataset(file_path)
    X = (df.index - df.index.min()).days.values.reshape(-1, 1)
    y = df["value"].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train_scaled, y_train)
    with open(save_path, 'wb') as f:
        pickle.dump((model, scaler), f)
    return save_path


def find_best_arima_params(file_path: str):
    train = load_dataset(file_path)
    train = train.asfreq('B')
    train['value'] = train['value'].fillna(method='ffill')

    smodel = auto_arima(train["value"],
                        start_p=0, start_q=0, max_p=3, max_q=3,
                        seasonal=False,  # <--- вот это
                        trace=False,
                        error_action='ignore', suppress_warnings=True,
                        stepwise=True)
    
    return {
        "order": smodel.order,
    }

def find_best_svr_params(file_path: str):
    df = load_dataset(file_path)
    X = (df.index - df.index.min()).days.values.reshape(-1, 1)
    y = df["value"].values
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }
    svr = SVR()
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_params_