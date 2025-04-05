import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd # работа с таблицами
import numpy as np # математические вычисления
from scipy import stats # статистические методы
import matplotlib
import matplotlib.pyplot as plt # визуализация данных

from dateutil.parser import parse # парсер даты

from statsmodels.tsa.seasonal import seasonal_decompose # оценка сезонности
from scipy.stats import normaltest # критерий Д'Агостино K^2, оценка данных на распределение Гаусса
from statsmodels.tsa.statespace.tools import diff  # разность рядов просто и/или сезонно вдоль нулевой оси
from statsmodels.tsa.arima_model import ARMAResults,ARIMAResults # Получить результаты после fit ARMA, ARIMA
from statsmodels.tsa.arima.model import ARIMA # Построить модель ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX # Построить модель SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # plot_acf график автокорреляции, plot_pacf частичной автокорреляции
from statsmodels.graphics.tsaplots import month_plot,quarter_plot # график сезонности данных по месяцам, кварталам
from pandas.plotting import lag_plot # график лага

from statsmodels.tsa.stattools import adfuller # тест Дики-Фуллера

from sklearn.metrics import mean_squared_error, accuracy_score # метрика качества MSE
from statsmodels.tools.eval_measures import rmse  # метрика качества Квадратный корень из MSE
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import GridSearchCV


from pmdarima import auto_arima # автоматический подбор параметров

def adf_test(series,title=''):
  '''
  тест Дики-Фуллера
  0 гипотеза: ряд данных не стационарен
  альтернативная гипотеза: ряд данных стационарен
  Понятие стационарного временного ряда означает, что его среднее значение не изменяется во времени, т. е. временной ряд не имеет тренда
  @param series - значения ряда
  @param title - заголовок ряда
  '''

  result = adfuller(series.dropna(),autolag='AIC')

  labels = ['ADF тест','p-value','# lags used','# наблюдения']
  out = pd.Series(result[0:4],index=labels)

  for key,val in result[4].items():
      out[f'критическое значение ({key})']=val

  print(out.to_string())

  if result[1] <= 0.05:
      print("Сильные доказательства против нулевой гипотезы")
      print("Отменяем 0 гипотезу")
      print("Данные стационарны")
  else:
      print("Слабые доказательства против нулевой гипотезы")
      print("Не отменяем 0 гипотезу")
      print("Данные не стационарны")

def poly_svr(df: pd.DataFrame):
    X = (df.index - df.index.min()).days.values.reshape(-1, 1)
    y = df["value"].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)) 

    model = SVR(kernel="poly", C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[len(X_train):], y_test, label="value", color="blue")
    plt.plot(df.index[len(X_train):], y_pred, label="pred", color="red")
    plt.legend()
    plt.title("Прогнозирование курса валют с помощью SVR")
    plt.show()

def rbf_svr(df: pd.DataFrame):
    X = (df.index - df.index.min()).days.values.reshape(-1, 1)
    y = df["value"].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)) 

    model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[len(X_train):], y_test, label="value", color="blue")
    plt.plot(df.index[len(X_train):], y_pred, label="pred", color="red")
    plt.legend()
    plt.title("Прогнозирование курса валют с помощью SVR")
    plt.show()

def get_arima_params(df: pd.DataFrame):
    TEST_SIZE = 600
    train= df.iloc[:len(df)-TEST_SIZE]
    train = train.asfreq('B')
    train['value'] = train['value'].fillna(method='ffill')
    test= df.iloc[len(df)-TEST_SIZE:]
    test = test.asfreq('B')
    test['value'] = test['value'].fillna(method='ffill')
    print(len(train))
    print(len(test))
    # найдем порядок p,d,q
    # ARIMA: обучение модели с сезонной составляющей
    smodel = auto_arima(train["value"],
                        start_p=0,
                        start_q=0,
                        max_p=3,
                        max_q=3,
                        m=20,
                        start_P=0,
                        seasonal=True,
                        d=None,
                        D=None,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)
    print(smodel.summary())
    with open('./arima.pkl', 'wb') as pkl:
        pickle.dump(smodel, pkl)
    #интерпретация ARIMA модели в виде графиков
    smodel.plot_diagnostics(figsize=(15,10))
    plt.show()

def arima_predict(df: pd.DataFrame):
    TEST_SIZE = 600
    train= df.iloc[:len(df)-TEST_SIZE]
    train = train.asfreq('B')
    train['value'] = train['value'].fillna(method='ffill')
    test= df.iloc[len(df)-TEST_SIZE:]
    test = test.asfreq('B')
    test['value'] = test['value'].fillna(method='ffill')
    print(len(train))
    print(len(test))
    #обучим модель
    sarima_model= SARIMAX(train["value"], order= (2,1,2), seasonal_order= (0,0,1,20), freq=train.index.inferred_freq)
    sarima_model_fit= sarima_model.fit()

    # сделаем пронозы на Test данных
    start= len(train)
    end= len(train) + len(test) -1
    pred= sarima_model_fit.predict(start=start, end=end, dynamic=False, typ="levels").rename("SARIMA predictions")

    # построим прогноз
    title= "Актуальность и прогноз для Test данных"
    test["value"].plot(title= title, legend=True)
    pred.plot(legend=True)
    plt.show()

    # оценим модель
    mse= mean_squared_error(test["value"],pred)
    rmse_sarima= rmse(test["value"], pred)

    print(f"RMSE= {rmse_sarima} \n")
    print(f"MSE= {mse}")



plt.rcParams["figure.figsize"] = (10,5) # размер графиков

plt.style.use('fivethirtyeight') # стиль графиков

# загрузка данных
df = pd.read_csv("./dataset.csv", index_col="date", parse_dates= True, sep=";")
df.index= pd.to_datetime(df.index)
df = df.sort_index(ascending=True)
df = df.asfreq('B')
df['value'] = df['value'].fillna(method='ffill')



# сконвертируем нестационарный ряд в стационарный
# Понятие стационарного временного ряда означает, что его среднее значение не изменяется во времени, т. е. временной ряд не имеет тренда
df["difference_1"]= diff(df["value"], k_diff=1)

adf_test(df["difference_1"])

poly_svr(df)
rbf_svr(df)
get_arima_params(df)
arima_predict(df)