import pandas as pd
import pickle
import numpy as np
from datetime import timedelta
from pathlib import Path
import os
import time

def predict(model_path: str, dataset_path: str, days: int) -> str:
    df = pd.read_csv(dataset_path, sep=';', parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    if Path(model_path).name.startswith("ARIMA"):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        df = df.set_index("date").asfreq("B")
        df["value"] = df["value"].fillna(method="ffill")

        start = len(df)
        end = start + days - 1
        forecast = model.predict(start=start, end=end, dynamic=False, typ="levels")

        result = pd.DataFrame({
            "date": pd.date_range(start=df.index[-1] + timedelta(days=1), periods=days, freq="B"),
            "value": forecast
        })

    elif Path(model_path).name.startswith("SVR"):
        with open(model_path, "rb") as f:
            model, scaler = pickle.load(f)

        df['days'] = (df['date'] - df['date'].min()).dt.days
        last_day = df['days'].max()
        future_days = np.arange(last_day + 1, last_day + days + 1).reshape(-1, 1)
        future_days_scaled = scaler.transform(future_days)

        predictions = model.predict(future_days_scaled)
        result = pd.DataFrame({
            "date": df['date'].max() + pd.to_timedelta(np.arange(1, days + 1), unit='D'),
            "value": predictions
        })

    else:
        raise ValueError("Unknown model type")

    output_path = f"./user_data/predictions/prediction_{os.path.basename(model_path[:-4])}_{int(time.time())}.csv"
    result.to_csv(output_path, sep=";", index=False)
    return output_path

def read_prediction(file_path: str) -> dict:
    df = pd.read_csv(file_path)
    return {
        "columns": df.columns.tolist(),
        "rows": df.to_dict(orient="records")
    }