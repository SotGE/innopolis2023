# -*- coding: utf-8 -*-
"""certification1-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JTtutB8c-qP5IczCOKSnzgl1ZWIOhHxi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import scipy
from scipy.stats import pearsonr
from scipy import stats

import sqlite3
import psycopg2

# Data Frame (загрузка CSV файла с данными разделенными запятой)
# Загрузка файла из Git репозитория в Pandas
df = pd.read_csv('https://raw.githubusercontent.com/SotGE/innopolis2023/main/certification1/dataset_tk.csv', sep=',')
# sns.load_dataset('titanic')

dataset = df.copy()

# Просмотр типов данных в датасете
dataset.info()

# Просмотр наименования колонок
# Дата
# Потребление электроэнергии в гигаваттах по штатам
print(dataset.columns) # list с заголовками
print(" ".join(dataset.columns))

# Первые ячейки
dataset.head(10)

# Последние ячейки
dataset.tail(10)

# Сумма нулевых значений по столбцам (проверяем датасет на наличие пропусков).
dataset.isnull().sum()

# Количество неопределенные значений (неправильно считанные)
dataset.isna().sum()

# Параметры числовых значений (фильтрация по числовым)
# С округлением до 2-х знаков после запятой
dataset.describe(include=['float', 'int']).round(2)

# Объем памяти DataFrame
memory = dataset.memory_usage(deep=True).sum()
print(f'Объем памяти, занимаемый DataFrame: {memory} байт')

# Кортеж из количества строк и столбцов
dtype = dataset.dtypes.value_counts()
print(f'Всего столбцов разных типов - {len(dtype)}')

# Кортеж из количества строк и столбцов
count = dataset.shape
print(f'Количество строк - {count[0]}, количество столбцов - {count[1]}')

# Удаление строк с пустыми занчениями
# Нет необходимости
# dataset = dataset.dropna()
# dataset

# Уникальные значения столбцов
nunique = dataset.nunique()
nunique

# Максимальные значения столбцов
maxi = dataset.max()
maxi

# Максимальные значения столбцов
mini = dataset.min()
mini

# Столбцы с интервальные переменными
categories_data = dataset.select_dtypes(include=['float64'])
# Столбцы в лист
categories = categories_data.columns.tolist()
print(f'Все выбранные столбцы: {categories}\n')
for category in categories:
  print(f'''
    Столбец [{category}]:
    Медиана: {categories_data[category].median()}
    Среднее: {categories_data[category].mean()}
    Максимум: {categories_data[category].max()}
    Минимум: {categories_data[category].min()}
    Квантиль 90: {categories_data[category].quantile(0.9)}
    Квантиль 10: {categories_data[category].quantile(0.1)}
    Колличество пропусков: {categories_data[category].isna().sum()}\n
  ''')

# Конвертация первого столбца даты в dtype: datetime64[ns]
# dataset['Unnamed: 0'] = pd.to_datetime(dataset['Unnamed: 0'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
# dataset['Unnamed: 0'] = dataset['Unnamed: 0'].astype('datetime64[ns]')
dataset.rename(columns={"Unnamed: 0":"Date"}, inplace=True)
dataset['Date']=pd.to_datetime(dataset["Date"], dayfirst=True)

# Настройка Date как Index
dataset.set_index('Date', inplace=True)

dataset

# Добавить отдельно столбцы: Год, Месяц, День
# dataset["Year"]=dataset["Date"].dt.year
# dataset["Month"]=dataset["Date"].dt.month
# dataset["Day"]=dataset["Date"].dt.day

# Удалить столбец: Дата
# dataset.drop(["Date"], axis=1, inplace=True)

# Потстроим матрицу корреляций
matrix_corr = dataset.corr()
matrix_corr

# Вывод корреляционной матрицы (тепловая карта)
heatmap = sns.heatmap(matrix_corr, annot=False, cmap='coolwarm')
heatmap.set_title('Корреляционная тепловая карта', fontdict={'fontsize':16}, pad=20);

# Потребление электроэнергии в гигаваттах по штатам в течение всего периода
dataset.plot(figsize=(20,16))
plt.xlabel('Дата')
plt.ylabel('Потребление электроэнергии')
plt.title('Потребление электроэнергии в гигаваттах по штатам')
plt.show()

# Среднее потребление электроэнергии в гигаваттах по штатам
power_mean = dataset.copy().mean()
power_mean.plot(figsize=(20,16))
plt.xlabel('Штаты')
plt.ylabel('Потребление электроэнергии')
plt.title('Среднее потребление электроэнергии в гигаваттах по штатам')
plt.show()

# Среднее потребление электроэнергии в гигаваттах по штатам
dataset_power_mean = dataset.copy().mean().reset_index()
plt.figure(figsize=(20,16))
sns.barplot(
  data = dataset_power_mean,
  x = "index",
  y = 0
)
plt.xticks(rotation=70, fontweight='light', fontsize='x-large')
plt.show()

# Минимальное потребление электроэнергии в гигаваттах по штатам
power_min = dataset.copy().min()
power_min.plot(figsize=(20,16))
plt.xlabel('Штаты')
plt.ylabel('Потребление электроэнергии')
plt.title('Минимальное потребление электроэнергии в гигаваттах по штатам')
plt.show()

# Минимальное потребление электроэнергии в гигаваттах по штатам
dataset_power_min = dataset.copy().min().reset_index()
plt.figure(figsize=(20,16))
sns.barplot(
  data = dataset_power_min,
  x = "index",
  y = 0
)
plt.xticks(rotation=70, fontweight='light', fontsize='x-large')
plt.show()

# Максимальное потребление электроэнергии в гигаваттах по штатам
power_max = dataset.copy().max()
power_max.plot(figsize=(20,16))
plt.xlabel('Штаты')
plt.ylabel('Потребление электроэнергии')
plt.title('Максимальное потребление электроэнергии в гигаваттах по штатам')
plt.show()

# Максимальное потребление электроэнергии в гигаваттах по штатам
dataset_power_max = dataset.copy().max().reset_index()
plt.figure(figsize=(20,16))
sns.barplot(
  data = dataset_power_max,
  x = "index",
  y = 0
)
plt.xticks(rotation=70, fontweight='light', fontsize='x-large')
plt.show()

"""### Гипотезы:
#### 1. В штате Punjab (штат Пенджаб) начиная с 2019 года растет потребление электроэнергии
#### 2. В штатах высокое потребление электроэнергии в зимний период за 2020 год
#### 3. В штатах низкое потребление электроэнергии в осенний период за 2020 год
"""

# Описательная статистика
# Отпределяем размер выборки для штата Punjab
n = dataset['Punjab']
n = 503

# Рассчитаем гауссовские распределенные данные
x = np.random.randn(n) + 2
y = np.random.randn(n)
var_x = x.var(ddof = 1)
var_y = y.var(ddof = 1)

# Рассчитаем стандартное отклонение
SD = np.sqrt((var_x + var_y) / 2)
print('Стандартное отклонение = ', SD)

# Вычисляем T-статистику
tval =(x.mean() - y.mean()) /(SD * np.sqrt(2 / n))

# Сравним ее с критическим значением T (для этого мы вычислили степени свободы и сравним значение p)
dof = 2 * n - 2

pval = 1 - stats.t.cdf(tval, df = dof)
print("t = " + str(tval))
print("p = " + str(2 * pval))

tval2, pval2 = stats.ttest_ind(x, y)
print("t = " + str(tval2))
print("p = " + str(pval2))

# Точечная диаграмма (рассеяния) потребления электроэнергии для штата Punjab

plt.figure(figsize=(20, 16))
plt.style.use('fivethirtyeight')

chart = sns.scatterplot(
  x=dataset.index,
  y='Punjab',
  data=dataset
)

chart.set(
  title='Точечная диаграмма (рассеяния) потребления электроэнергии для штата Punjab',
  ylabel='# Потребления электроэнергии',
  xlabel="Дата"
)

# Линия тренда
z = np.polyfit(x, dataset['Punjab'], 1)
p = np.poly1d(z)
plt.plot(dataset.index, p(x), c="b", ls=":")

# Линейная диаграмма потребления электроэнергии для штата Punjab
# Линия тренда
sns.lineplot(
  data = dataset,
  x = "Date",
  y = "Punjab"
)
plt.xticks(rotation=45)

"""#### В штате Punjab потребление электроэнергии не растет, зато растет максимальная мощность электроэнергии"""

# Выборка данных за 2020 год
dataset_interval = dataset.copy().reset_index()
start_date = '2020-01-01'
end_date = '2020-12-31'
dataset_interval = dataset_interval.loc[dataset_interval['Date'].between(start_date, end_date)]

# Настройка Date как Index
dataset_interval.set_index('Date', inplace=True)

dataset_interval

# Потребление электроэнергии в гигаваттах по штатам за 2020 год
dataset_interval.plot(figsize=(20,16))
plt.xlabel('Дата')
plt.ylabel('Потребление электроэнергии')
plt.title('Потребление электроэнергии в гигаваттах по штатам')
plt.show()

# Среднее потребление электроэнергии в гигаваттах по штатам за 2020 год
# С сортировкой
dataset_mean = dataset_interval.mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(20,16))
sns.barplot(
  data = dataset_mean,
  x = "index",
  y = 0
)
plt.xticks(rotation=70, fontweight='light', fontsize='x-large')
plt.show()

# Минимальное потребление электроэнергии в гигаваттах по штатам за 2020 год
# С сортировкой
dataset_min = dataset_interval.min().sort_values(ascending=False).reset_index()
plt.figure(figsize=(20,16))
sns.barplot(
  data = dataset_min,
  x = "index",
  y = 0
)
plt.xticks(rotation=70, fontweight='light', fontsize='x-large')
plt.show()

# Максимальное потребление электроэнергии в гигаваттах по штатам за 2020 год
# С сортировкой
dataset_max = dataset_interval.max().sort_values(ascending=False).reset_index()
plt.figure(figsize=(20,16))
sns.barplot(
  data = dataset_max,
  x = "index",
  y = 0
)
plt.xticks(rotation=70, fontweight='light', fontsize='x-large')
plt.show()

# Показатель потребления электроэнергии по штату Punjab
dataset_punjab = dataset_interval.copy().reset_index()
plt.figure(figsize=(20,16))
plt.plot(dataset_punjab['Date'],dataset_punjab['Punjab'])

plt.figure(figsize=(20,16))
plt.title('Потребления электроэнергии по штату Punjab')
plt.hist(dataset_punjab['Punjab'])

# Помесячные данные потребление электроэнергии сгруппированные по штатам за 2020 год
month = dataset_interval.copy().reset_index()
month = month.groupby(month['Date'].dt.strftime('%B')).sum(numeric_only=True)
month

# Помесячные данные потребление электроэнергии сгруппированные по штату Punjab за 2020 год
month['Punjab']

# Потребление электроэнергии по штатам за каждый месяц 2020 года
plt.figure(figsize=(20,16))
plt.title('Потребление электроэнергии по штатам за каждый месяц 2020 года')
sns.heatmap(month, annot=False, cmap='coolwarm')

# Сравнение 2-х штатов по потреблению электроэнергии за 2020 год (помесячно)
plt.figure(figsize=(20,16))
plt.plot(month['Punjab'], marker = 'o')
plt.plot(month['Gujarat'], marker = 's')
plt.xlabel('Месяц')
plt.ylabel('Месячное потребление электроэнергии')
plt.title("Сравнение 2-х штатов по потреблению электроэнергии за 2020 год (помесячно)")
plt.legend(['Punjab' , 'Gujarat'])

# Сравнение всех штатов по потреблению электроэнергии за 2020 год (помесячно)
plt.figure(figsize=(20,16))
plt.plot(month, marker = 'o')
plt.xlabel('Месяц')
plt.ylabel('Месячное потребление электроэнергии')
plt.title("Сравнение 2-х штатов по потреблению электроэнергии за 2020 год (помесячно)")
plt.legend(month)

"""### За 2020 год
#### С января по апрель, потребление электроэнергии высокое
#### Июнь, июль, август, сентября, октябрь и декабрь, низкое потребление электроэнергии

#### В середине зимнего периода январь-февраль приходится большое потребление электроэнергии, скорее всего связано это с тем, что в это время наплыв туристов в 2020 году
#### Начиная с июня, приходится низкое потребление электроэнергии в 2020 году по конец года
"""

# Построение графиков

# Фигура 900 на 600 пикселей
fig = plt.figure(figsize=(20, 16))
# Распределение оси Х, шаг от 0 до 80 (10 точек)
min = 0
max = 80
counter = 10
x_points = np.linspace(min, max, counter)

# Гистограмма
def show_plot(data, x_points, color, label, xylabel, title):
    '''
    Функция для отображения гистограммы
    :param data - данные, массив столбцов
    :param x_points - массив точек оси Х
    :param color - массив цветов для графиков
    :param label - подписи легенда
    :param xylabel - подписи осей
    :param title - заголовок
    '''
    plt.hist(data, color=color, label=label)
    # расположение легенды (справа вверху)
    plt.legend(loc='upper right')
    plt.xlabel(xylabel[0])
    plt.ylabel(xylabel[1])
    plt.title(title)
    plt.show()

# Разделение выборки на каждый месяц
month = month.reset_index()

isApril = month['Date'] == 'April'
isAugust = month['Date'] == 'August'
isDecember = month['Date'] == 'December'
isFebruary = month['Date'] == 'February'
isJanuary = month['Date'] == 'January'
isJuly = month['Date'] == 'July'
isJune = month['Date'] == 'June'
isMarch = month['Date'] == 'March'
isMay = month['Date'] == 'May'
isNovember = month['Date'] == 'November'
isOctober = month['Date'] == 'October'
isSeptember = month['Date'] == 'September'

datasetApril = month.loc[isApril]
datasetAugust = month.loc[isAugust]
datasetDecember = month.loc[isDecember]
datasetFebruary = month.loc[isFebruary]
datasetJanuary = month.loc[isJanuary]
datasetJuly = month.loc[isJuly]
datasetJune = month.loc[isJune]
datasetMarch = month.loc[isMarch]
datasetMay = month.loc[isMay]
datasetNovember = month.loc[isNovember]
datasetOctober = month.loc[isOctober]
datasetSeptember = month.loc[isSeptember]

print(datasetApril)

# Влияние штата Punjab на январь (высокое потребление электроэнергии) и август (низкое потребление электроэнергии)
data_to_plot = [
  datasetApril['Punjab'],
  datasetAugust['Punjab'],
  datasetDecember['Punjab'],
  datasetFebruary['Punjab'],
  datasetJanuary['Punjab'],
  datasetJuly['Punjab'],
  datasetJune['Punjab'],
  datasetMarch['Punjab'],
  datasetMay['Punjab'],
  datasetNovember['Punjab'],
  datasetOctober['Punjab'],
  datasetSeptember['Punjab']
]
colors = [
  'gray',
  'blue',
  'red',
  'silver',
  'brown',
  'aqua',
  'green',
  'plum',
  'crimson',
  'indigo',
  'teal',
  'navy'
]
labels = [
  'April',
  'August',
  'December',
  'February',
  'January',
  'July',
  'June',
  'March',
  'May',
  'November',
  'October',
  'September'
]
xylabel = ['Штат Punjab ', 'Потребление электроэнергии']
title = 'Статистика потребление электроэнергии штата Punjab в 2020 году'
show_plot(
    data=data_to_plot,
    x_points=x_points,
    color=colors,
    label=labels,
    xylabel=xylabel,
    title=title
)