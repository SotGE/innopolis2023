# %%
# py -m pip install --upgrade pip setuptools wheel
# py -m pip install catboost
# py -m pip install ipywidgets
# py -m pip install scikit-learn

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

from sklearn.preprocessing import LabelEncoder # Кодирование категориальных данных
from sklearn.feature_selection import SelectKBest # Выбор признаков
from sklearn.feature_selection import chi2 # Выбор признаков по Хи квадрат

from sklearn.model_selection import train_test_split # Деление выборки на тестовые и тренировочные данные

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # Критерий качества, точности

# %%
# Генерируем тестовые данные и сохраняем в CSV файл
def generate_students(subjects: int, count: int):
  objects = [f"subject_{num + 1}" for num in range(subjects)]
  grades = np.random.randint(0, 100, (count, subjects))
  data = pd.DataFrame(data = grades, columns = objects)
  data.sum(axis = 0)
  data['score_mean'] = np.round((data.sum(axis = 1) / subjects), 3)
  for index, row in data.iterrows():
    if (row['score_mean'] < 45):
      data.loc[index, 'score_text'] = 'удовлетворительно'
    elif (row['score_mean'] >= 45 and row['score_mean'] < 75):
      data.loc[index, 'score_text'] = 'хорошо'
    else:
      data.loc[index, 'score_text'] = 'отлично'
  return data

generate = generate_students(subjects = 6, count = 2000)
generate.to_csv('data.csv', index = False)
generate


# %%
# Загружаем данные из CSV файла
data = pd.read_csv('data.csv')
data

# %%
# Определение признаков
X = data.drop(['score_mean', 'score_text'], axis = 1)
X

# %%
# Определение целевой переменной
y = data['score_text']
y

# %%
# Преобразовываем категориальный признак в числовой
label_encoder = LabelEncoder()
temp_y_encoded = label_encoder.fit_transform(y)
temp_y_encoded

# %%
# Объеденяем категориальный признак текстовой с числовым и сохраняем в CSV файл
temp_y = y.values
y_new = pd.DataFrame(data = list(zip(temp_y, temp_y_encoded)), columns = ['y', 'y_encoded'])
y_new.to_csv('score_text_labels.csv', index=False)
y_new

# %%
# Выбираем лучшие 3 признака с использованием критерия хи-квадрат
y = temp_y_encoded
selector = SelectKBest(chi2, k = 3)
X_best = selector.fit_transform(X, y)
X_best

# %%
# Разделяем датасет на тестовую и обучающую выборку
# 80% обучающей выборки, 20% тестовой
X = X_best
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 64)

# %%
# Обучаем модель CatBoostClassifier и сохраняем модель в файл
model = CatBoostClassifier(iterations = 100, depth = 4, learning_rate = 0.1)
model.fit(X_train, y_train)
model.save_model('people_university_model_eff.cbm')

# %%
# Загружаем мобель из файла
model_new = CatBoostClassifier()
model_new.load_model('people_university_model_eff.cbm')

# %%
# Делаем предсказания на тестовом наборе
y_pred = model_new.predict(X_test)
y_pred

# %%
# Оцениваем точность модели
accuracy_eff = accuracy_score(y_test, y_pred)
print(f"Правильность (accuracy) модели: {accuracy_eff}")

precision_eff = precision_score(y_test, y_pred, average='weighted')
print(f"Точность (precision) модели: {precision_eff}")

recall_eff = recall_score(y_test, y_pred, average='weighted')
print(f"Полнота (recall) модели: {recall_eff}")

f1_eff = f1_score(y_test, y_pred, average='weighted')
print(f"F1 мера модели: {f1_eff}")

# %%
def classifyStudent(grades_new):
  # Преобразование данных студента в DataFrame
  data_new = pd.DataFrame([grades_new])

  # Прогнозирование с использованием обученной модели (SelectBestK)
  predicted_category = model_new.predict(data_new)[0][0]

  # Преобразование обратно в текстовую категорию
  categories = ['отлично', 'удовлетворительно', 'хорошо']
  predict = categories[predicted_category]
  return predict


# %%
# Генерируем данные для прогноза
grades_new = np.random.randint(0, 100, 6)
grades_new

# %%
# Прогнозирование
predict = classifyStudent(grades_new)
print(f"Студент скорее всего получит (SelectBestK): {predict}")


