import requests
import time
import re
from pathlib import Path
import pandas as pd

valutes = [
    "AUD","AZN","GBP","AMD",
    "BYN","BGN","BRL","HUF",
    "VND","HKD","GEL","DKK",
    "AED","USD","EUR","EGP",
    "INR","IDR","KZT","CAD",
    "QAR","KGS","CNY","MDL",
    "NZD","NOK","PLN","RON",
    "XDR","SGD","TJS","THB",
    "TRY","TMT","UZS","UAH",
    "CZK","SEK","CHF","RSD",
    "ZAR","KRW","JPY",
]
      
def create_dataset(valute, days, max_retries=3, output_dir: str = "./user_data/datasets", time_delay=0.7):
    current_day = 1
    retries = 0
    https = "https:"
    current_url = "//www.cbr-xml-daily.ru/daily_json.js"
    file_path = f"{output_dir}/{valute}_dataset_{int(time.time())}.csv"
    result_file = open(file_path, "w+", encoding="utf-8")
    result_file.write("date;value\n")
    try:
        while (current_day <= days):
            try:
                raw_data = requests.get(https + current_url, timeout=30)
                raw_data.raise_for_status()
            except Exception as e:
                if __name__ == '__main__':
                    print("Что-то пошло не так:\n" + e.__class__.__name__)
                if retries < max_retries:
                    if __name__ == '__main__':
                        print("Повторяем запрос. Попытка #" + str(retries))
                    retries += 1
                    time.sleep(5)
                    continue
                else:
                    result_file.close()
                    return False
            retries = 0
            result_JSON = raw_data.json()
            current_url = result_JSON["PreviousURL"]
            value = result_JSON["Valute"][str(valute)]["Value"]
            date = re.match(pattern="(.*)T.*", string=result_JSON["Date"]).group(1)
            result_file.write(str(date) + ";" + str(value) + "\n")
            if __name__ == '__main__':
                print("День #" + str(current_day) + ". дней осталось: " + str(days - current_day) + ". дата: "+ str(date) + ". стоимость: " + str(value) + "₽\n" )
            time.sleep(time_delay)
            current_day += 1
        result_file.close()
    except KeyboardInterrupt:
        if __name__ == '__main__':
            print("Выполнение скрипта прервано досрочно. Файл сохранён.")
        result_file.close()
        return file_path
    return file_path


def read_dataset(file_path: str) -> dict:
    df = pd.read_csv(file_path)
    return {
        "columns": df.columns.tolist(),
        "rows": df.to_dict(orient="records")
    }

def group_dataset_by_month(original_path: str, save_dir: str = "./user_data/datasets"):
    df = pd.read_csv(original_path, sep=';', parse_dates=['date'])
    monthly_df = (
        df
        .set_index('date')
        .resample('MS')
        .mean()
        .reset_index()
    )

    monthly_df['date'] = monthly_df['date'].dt.strftime('%Y-%m-%d')

    original_name = Path(original_path).stem
    new_filename = f"{original_name}_monthly.csv"
    new_path = Path(save_dir) / new_filename
    monthly_df.to_csv(new_path, sep=';', index=False)

    return str(new_path)



def main():
    print("Введите код валюты (USD): ", end="")
    valute = str(input()).upper()
    if valute.strip() == "":
        valute = "USD"
    if not (valute in valutes):
        print("Введённая валюта не поддерживается")
        return
    
    print("Введите количество дней (1): ", end="")
    days = input()
    if days.strip() == "":
        days = 1
    else:
        try:
            int(days.strip())
        except ValueError:
            print("Введено не целое число")
            return
        days = int(days.strip())
        if(days < 1):
            print("Количество дней должно быть больше нуля")
            return
    
    print("Введите количество повторных попыток в случае ошибки (3): ", end="")
    max_retries = input()
    if max_retries.strip() == "":
        max_retries = 3
    else:
        try:
            int(max_retries.strip())
        except ValueError:
            print("Введено не целое число")
            return
        max_retries = int(max_retries.strip())
        if(max_retries < 0):
            print("Количество попыток должно должно быть положительным")
            return
    
    print("Введите интервал между запросами в секундах (0.7): ", end="")
    time_delay = input()
    if time_delay.strip() == "":
        time_delay = 0.7
    else:
        try:
            float(time_delay.strip())
        except ValueError:
            print("Введено не число")
            return
        time_delay = float(time_delay.strip())
        if(time_delay < 0):
            print("Интервал между запросами должен быть положительным")
            return
        
    if create_dataset(valute=valute, days=days, max_retries=max_retries, time_delay=time_delay):
        print("Датасет успешно создан")
    else:
        print("При создании датасета возникла ошибка")
    
if __name__ == '__main__':
    main()