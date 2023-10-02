import gdown

import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder # Кодирование категориальных данных

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler # Масштабирование данных

from sklearn.feature_selection import SelectKBest # Выбор признаков с наивысшими оценками
from sklearn.feature_selection import chi2 # Выбор признаков по Хи квадрат

from sklearn.model_selection import train_test_split # Деление выборки на тестовые и тренировочные данные
from sklearn.model_selection import cross_val_score # Оценка качества работы модели

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # Критерий качества, точности

from sklearn.neighbors import KNeighborsClassifier # Обучение модели K-ближайших соседей
from sklearn.linear_model import LinearRegression # Линейная регрессия (метод наименьших квадратов)

from sklearn.tree import DecisionTreeClassifier # Деревья решений
from sklearn.ensemble import RandomForestClassifier # Ансамбли деревьев решений
from sklearn.ensemble import GradientBoostingClassifier # Ансамбли градиентного спуска

from sklearn.ensemble import RandomForestRegressor # случайный лес
from sklearn.neighbors import KNeighborsRegressor # метод ближайших соседей
from sklearn.svm import SVR # метод опорных векторов с линейным ядром
from sklearn.linear_model import LogisticRegression # логистическая регрессия

from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif # Статистический метод
from sklearn.metrics import r2_score

# %%
# Скачивание данных из Google Disk
# gdown.download(id='1LBDnhITL0Wqwp5G6M6IBI-SSz8BIoNec')

# Загрузка файла из Git репозитория в Pandas
# dataset = pd.read_csv('https://raw.githubusercontent.com/SotGE/innopolis2023/main/lesson12/diabetes.csv', sep=',')

# Загрузка данных из локального хранилища
dataset = pd.read_csv(r"diabetes.csv", sep=',')

# Первые ячейки
dataset.head(5)

# %%
# Размер данных (количество строк, колонок)
dataset.shape

# %%
# Заголовки столбцов в нижнем регистре
dataset.columns = [col.lower() for col in dataset.columns]
dataset.columns

# %%
# Проверка пропущенных значений
dataset.isnull().mean()

# %%
# Количество неопределенные значений (неправильно считанные)
dataset.isna().mean()

# %%
# Проверка значений на 0
(dataset == 0).sum()

# %%
# Заполнение нулевых значений - медианой
dataset = dataset.replace(0, dataset.median())
dataset

# %%
# Проверка значений на 0
(dataset == 0).sum()

# %%
# Описательная статистика
dataset.describe(include='all', percentiles=[0.1, 0.25,0.5, 0.75, 0.9]).T

# %%
# Просмотр типов данных в датасете
dataset.info()

# %%
# Разделение для задачи классификации на X (data features) и y (outcome)
X = dataset.drop(columns=['outcome'])
y = dataset['outcome']

# %%
# Построить распределение для всех числовых переменных
figure = px.box(X)
figure.show()

# %%
# Подготовка данных
# Нормализация (StandardScaler)
scalar = StandardScaler()
features = scalar.fit_transform(X, y)
features

# %%
# Массив в Pandas
X_normalised = pd.DataFrame(features, columns=X.columns)
X_normalised

# %%
# Построить распределение для всех числовых нормализированных переменных
figure = px.box(X_normalised)
figure.show()

# %%
# Разделение на тренировочную и тестовую (25%) для классификации
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=512, shuffle=True)

# %%
# Укажите score модели (метрики) и постройте визуализацию обученной классификации для указанных k

def model_report(model, X_test, y_test, average='weighted'):
  # Проведите тестирование модели
  # Делаем предсказания на тестовом наборе
  y_pred = model.predict(X_test)

  # Оцениваем точность модели
  accuracy_eff = accuracy_score(y_test, y_pred)
  print(f"Правильность (accuracy) модели: {accuracy_eff}")

  precision_eff = precision_score(y_test, y_pred, average=average)
  print(f"Точность (precision) модели: {precision_eff}")

  recall_eff = recall_score(y_test, y_pred, average=average)
  print(f"Полнота (recall) модели: {recall_eff}")

  f1_eff = f1_score(y_test, y_pred, average=average)
  print(f"F1 мера модели: {f1_eff}")
  
  # # Построение графика решающих областей 2D для бинарной классификации
  # x_min, x_max = X_test.to_numpy()[:, 0].min() - 1, X_test.to_numpy()[:, 0].max() + 1
  # y_min, y_max = X_test.to_numpy()[:, 1].min() - 1, X_test.to_numpy()[:, 1].max() + 1
  # xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

  # # Предсказание значений на сетке
  # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  # Z = Z.reshape(xx.shape)

  # # Построение контуров
  # plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
  
  # Разметка классов
  for i, c in zip(range(2), 's^o'):
      plt.scatter(X_test.to_numpy()[y_test.to_numpy() == i][:, 0], X_test.to_numpy()[y_test.to_numpy() == i][:, 1], marker=c, label=f"Class {i}")

  plt.xlabel('Признак 1')
  plt.ylabel('Признак 2')
  plt.title(f'Результат классификации ({type(model).__name__})')
  plt.legend()
  plt.show()

# %%
# Загрузить классификатор модели k-Nearest Neighbors (kNN - k-ближайших соседей)
model_knn = KNeighborsClassifier()

# %%
# Построение модели k-Nearest Neighbors (kNN - k-ближайших соседей)
model_knn.set_params(n_neighbors=25)
model_knn.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(model_knn.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(model_knn.score(X_test, y_test)))
model_report(model_knn, X_train, y_train)

# %%
# Рассчитайте модель kNN для k = 5, 10, 15, 20, 25
score_best = 0
best_n_neighbors = 0

for k in np.arange(5, 26, 5):
  model_knn.set_params(n_neighbors=k)
  scores = cross_val_score(model_knn, X_train, y_train)
  print(f"k: {k}, по scores - {scores}")
  score_avg = np.mean(scores)
  if score_avg > score_best:
    score_best = score_avg
    best_n_neighbors = k

print(f"Лучший k по scores - {best_n_neighbors}")

# %%
# Построение модели k-Nearest Neighbors (kNN - k-ближайших соседей)
model_knn_best = KNeighborsClassifier(n_neighbors=5)
model_knn_best.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(model_knn_best.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(model_knn_best.score(X_test, y_test)))
model_report(model_knn_best, X_train, y_train)

# %%
# Выбор 2 лучших признаков по Хи квадрат
selector = SelectKBest(chi2, k=2)

X_train_best = selector.fit_transform(X_train, y_train)
X_train_best_df = pd.DataFrame(X_train_best)
X_train_best_df

X_test_best = selector.fit_transform(X_test, y_test)
X_test_best_df = pd.DataFrame(X_test_best)
X_test_best_df

# %%
# Загрузить классификатор модели k-Nearest Neighbors (kNN - k-ближайших соседей)
model_knn_chi2 = KNeighborsClassifier()

# %%
# Построение модели k-Nearest Neighbors (kNN - k-ближайших соседей)
model_knn_chi2.set_params(n_neighbors=25)
model_knn_chi2.fit(X_train_best_df, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(model_knn_chi2.score(X_train_best_df, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(model_knn_chi2.score(X_test_best_df, y_test)))
model_report(model_knn_chi2, X_train_best_df, y_train)

# %%
# Рассчитайте модель kNN для k = 5, 10, 15, 20, 25
score_best = 0
best_n_neighbors = 0

for k in np.arange(5, 26, 5):
  model_knn_chi2.set_params(n_neighbors=k)
  scores = cross_val_score(model_knn_chi2, X_train_best_df, y_train)
  print(f"k: {k}, по scores - {scores}")
  score_avg = np.mean(scores)
  if score_avg > score_best:
    score_best = score_avg
    best_n_neighbors = k

print(f"Лучший k по scores - {best_n_neighbors}")

# %%
# Построение модели k-Nearest Neighbors (kNN - k-ближайших соседей)
model_knn_chi2_best = KNeighborsClassifier(n_neighbors=10)
model_knn_chi2_best.fit(X_train_best_df, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(model_knn_chi2_best.score(X_train_best_df, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(model_knn_chi2_best.score(X_test_best_df, y_test)))
model_report(model_knn_chi2_best, X_train_best_df, y_train)

# %%
# Работа с множественной линейной регрессией
# {SkinThickness, BMI} и Y = {Insulin} из датасета

X_regr = dataset[['skinthickness', 'bmi']]
y_regr = dataset['insulin']

# %%
# График зависимости skinthickness и insulin
sns.scatterplot(x=dataset['skinthickness'], y=dataset['insulin'])

# %%
# График зависимости bmi и insulin
sns.scatterplot(x=dataset['bmi'], y=dataset['insulin'])

# %%
# График коррелиции зависимости bmi и insulin
dataset['mean'] = (dataset['bmi'] + dataset['skinthickness']) / 2
sns.scatterplot(x=dataset['mean'], y=dataset['insulin'])

# %%
# Разделение на тренировочную и тестовую (25%) для классификации
X_regr = dataset[['mean']]
X_train, X_test, y_train, y_test = train_test_split(X_regr, y_regr, test_size=0.25, random_state=512, shuffle=True)

# %%
# Тренируем модель линейной регрессией
model_regr = LinearRegression()
model_regr.fit(X_train, y_train)

# %%
# Метрика качества линейной регрессией на тестовой выборке
model_regr.score(X_test, y_test)

# %%
# Прогноз значения целевой функции y от x
y_pred = model_regr.predict(X_test)

# %%
# Вывод предсказание, реальное значение в тестовой выборке и разница между ними
d = {
  'y_pred': y_pred,
  'y_test': y_test,
  'diff': np.abs(y_test - y_pred)
}
pd.DataFrame(d)
# Линейная регрессия не подходит под текущие данные

# %%
# Работа с алгоритмом DecisionTreeClassifier (используем ансамбли деревьев решений)
tree = DecisionTreeClassifier(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=512, shuffle=True)
tree.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(tree.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(tree.score(X_test, y_test)))
# Мы не указывали никаких параметров у этой модели. По умолчанию, дерево растет до тех пор, пока все листья не станут "чистыми"
# и дает 100% правильность на обучающем наборе. На тестовом, как мы видим, есть ошибки. Найдем глубину дерева.

# %%
tree.get_depth()

# %%
# Ограничим дерево по глубине. Это может улучшить точность на тестовых данных. Точность на тренировочном наборе при этом падает.
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(tree.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(tree.score(X_test, y_test)))

# %%
# Похожие результаты дают ансамбли деревьев решений, хотя, учитывая множество параметров у этих методов,
# их можно постараться настроить и получше.
rf = RandomForestClassifier(min_samples_split=5, n_estimators=1000)
rf.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(rf.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(rf.score(X_test, y_test)))

# %%
gb = GradientBoostingClassifier(max_depth=1, n_estimators=1000)
gb.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(gb.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(gb.score(X_test, y_test)))

# %%
# Рассмотрим метрики разных алгоритмов
models = [LinearRegression(), # метод наименьших квадратов
          RandomForestRegressor(n_estimators=100, max_features ='sqrt'), # случайный лес
          KNeighborsRegressor(n_neighbors=6), # метод ближайших соседей
          SVR(kernel='linear'), # метод опорных векторов с линейным ядром
          LogisticRegression() # логистическая регрессия
          ]

# Для каждой модели из списка
for model in models:
  # Обучаем модель
  model.fit(X_train, y_train)
  print(f"\nМодель {type(model).__name__}:")
  print("Правильность на обучающем наборе: {:.3f}".format(model.score(X_train, y_train)))
  print("Правильность на тестовом наборе: {:.3f}".format(model.score(X_test, y_test)))

# %%
# Деревья решений также дают возможность оценить важность признаков в рассматриваемой задаче.
# Это число варьирует в диапазоне от 0 до 1 для каждого признака, где 0 означает «не используется вообще»,
# а 1 означает, что «отлично предсказывает целевую переменную».
# Для задач, подобно нашей, с малым числом признаков это не очень интересно,
# тем не менее давайте посмотрим и сравним с тем результатом, который выдает статистический метод GenericUnivariateSelect.
print("\nВажности признаков (деревьев решений):\n{}".format(tree.feature_importances_))
print("\nВажности признаков (ансамбли деревьев решений):\n{}".format(rf.feature_importances_))

# %%
importances = pd.Series(rf.feature_importances_)
importances.plot.bar()

# %%
selector=GenericUnivariateSelect(score_func=mutual_info_classif, mode='k_best', param=3)
selector.fit(X_train, y_train)
pd.DataFrame(data={'score':selector.scores_,
                   'support':selector.get_support()}).sort_values(by='score', ascending=False)


