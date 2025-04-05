import pandas as pd
import matplotlib.pyplot as plt

def create_graph_whole_period(df: pd.DataFrame):
    """Функция для построения графика изменения курса за весь период.
    Args:
        df: DataFrame, содержащий столбцы 'date' и 'value'.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['value'], linestyle='-', color='b', label='Value')

    plt.title('Изменение курса за весь период', fontsize=16)
    plt.xlabel('Дата', fontsize=14)
    plt.ylabel('Курс', fontsize=14)
    plt.xticks(rotation=45)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def create_graph_month(df: pd.DataFrame, month:str):
    """Функция для построения графика изменения курса за указанный месяц с медианой и средним значением.
    Args:
        df: DataFrame, содержащий столбцы 'date' и 'value'.
        month: Месяц в формате 'YYYY-MM' для фильтрации данных.
    """

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    filtered_df = df[df['date'].dt.to_period('M') == month]

    median_value = filtered_df['value'].median()
    mean_value = filtered_df['value'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df['date'], filtered_df['value'], marker='o', linestyle='-', color='b', label='Курс')

    plt.axhline(y=median_value, color='g', linestyle='--', label='Медиана')
    plt.axhline(y=mean_value, color='r', linestyle='--', label='Среднее значение')

    plt.title(f'Изменение курса за {month}', fontsize=16)
    plt.xlabel('Дата', fontsize=14)
    plt.ylabel('курс', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def calculate_mean_by_month(df):
    """Функция для вычисления среднего значения курса за месяц.
    Args:
        df: DataFrame, содержащий столбцы 'date' и 'value'.
    Returns:
        DataFrame, сгруппированный по месяцу, с рассчитанным средним значением курса.
    """

    df['month'] = df['date'].dt.to_period('M')

    monthly_mean = df.groupby('month')['value'].mean().reset_index()

    return monthly_mean


def filter_by_deviation(df:pd.DataFrame, threshold: float):
    """Функция для фильтрации DataFrame по отклонению от среднего значения курса.
    Args:
        df: DataFrame
        threshold: Значение отклонения от среднего значения
    Returns:
        Отфильтрованный DataFrame.
    """

    if 'deviation_from_mean' not in df.columns:
        raise ValueError("DataFrame должен содержать столбец 'deviation_from_mean'.")
    
    filtered_df = df[df['deviation_from_mean'] >= threshold]
    
    return filtered_df

def filter_by_date(df: pd.DataFrame, start_date: str, end_date: str):
    """Функция для фильтрации DataFrame по диапазону дат.
    Args:
        df: DataFrame
        start_date: Начальная дата в формате 'YYYY-MM-DD'.
        end_date: Конечная дата в формате 'YYYY-MM-DD'.
    Returns:
        Отфильтрованный DataFrame.
    """
    
    # Преобразуем столбец 'date' в тип datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Фильтрация строк, где дата находится в заданном диапазоне
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    return filtered_df

def main():
    print("Введите путь до файла (./dataset.csv): ", end="")
    filename = input()
    if filename.strip() == "":
        filename = "./dataset.csv"

    print("Введите значение отклонения (10.0): ", end="")
    threshold = input()
    if threshold.strip() == "":
        threshold = 10.0
    else:
        try:
            float(threshold.strip())
        except ValueError:
            print("Введено не число")
            return
        threshold = float(threshold.strip())

    print("Введите начальную дату (2024-06-11): ", end="")
    start_date = input()
    if start_date.strip() == "":
        start_date = "2024-06-11"
    
    print("Введите конечную дату (2024-07-11): ", end="")
    end_date = input()
    if end_date.strip() == "":
        end_date = "2024-07-11"
    
    print("Введите месяц (2024-03): ", end="")
    month = input()
    if month.strip() == "":
        month = "2024-03"
    
    df = pd.read_csv(filename, sep=";")
    df.columns = ["date", "value"]
    df.dropna(subset=["date"], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.fillna({"value": df["value"].mean()}, inplace=True)

    median_value = df['value'].median()
    mean_value = df['value'].mean()

    df['deviation_from_median'] = df['value'] - median_value
    df['deviation_from_mean'] = df['value'] - mean_value

    stats = df[['value', 'deviation_from_median', 'deviation_from_mean']].describe()
    print("Датафрейм с новыми столбцами")
    print(df)
    print("Describe")
    print(stats)
    print("Фильтр по отклонению")
    print(filter_by_deviation(df=df,threshold=threshold))
    print("Фильтр по дате")
    print(filter_by_date(df=df, start_date=start_date, end_date=end_date))
    print("Среднее значение за месяц")
    print(calculate_mean_by_month(df=df))
    create_graph_whole_period(df=df)
    create_graph_month(df=df, month=month)

if __name__ == '__main__':
    main()