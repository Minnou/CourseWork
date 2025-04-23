import pandas as pd
import pickle
import numpy as np
from datetime import timedelta
from pathlib import Path
import os
import time

def predict(model_path: str, dataset_path: str, days: int, window_size: int = 30) -> str:
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

        values = df["value"].values
        if len(values) < window_size:
            raise ValueError(f"Not enough data for window of size {window_size}")

        last_window = values[-window_size:]
        predictions = []
        current_window = last_window.copy()

        for _ in range(days):
            scaled_window = scaler.transform(current_window.reshape(1, -1)).flatten()
            prediction = model.predict([scaled_window])[0]
            predictions.append(prediction)
            current_window = np.append(current_window[1:], prediction)

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