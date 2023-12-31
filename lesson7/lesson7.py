# -*- coding: utf-8 -*-
"""lesson7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UHtmoSZ8u9gMubm0TVjRMgXedVFgCynF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sqlite3
import psycopg2

# from google.colab import drive
# drive.mount('/content/drive')

# Добавление таблиц в sqlite3
conn = sqlite3.connect('titanic.db')
print('Подключение установлено')
# conn.autocommit = True  # устанавливаем актокоммит
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE embarked (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embarked TEXT NOT NULL
);
''')
cursor.execute('''
CREATE TABLE cabin (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cabin TEXT NOT NULL
);
''')
cursor.execute('''
CREATE TABLE pclass (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pclass INTEGER NOT NULL,
    cabin_id INTEGER,
    FOREIGN KEY (cabin_id) REFERENCES cabin (id)
);
''')
cursor.execute('''
CREATE TABLE ticket (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pclass_id INTEGER,
    cabin_id INTEGER,
    embarked_id INTEGER,
    ticket TEXT NOT NULL,
    fare REAL NOT NULL,
    FOREIGN KEY (pclass_id) REFERENCES pclass (id),
    FOREIGN KEY (cabin_id) REFERENCES cabin (id),
    FOREIGN KEY (embarked_id) REFERENCES embarked (id)
);
''')
cursor.execute('''
CREATE TABLE passenger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    survived INTEGER NOT NULL,
    name TEXT NOT NULL,
    sex TEXT NOT NULL,
    age REAL NOT NULL,
    sibSp INTEGER,
    parch INTEGER,
    ticket_id INTEGER,
    FOREIGN KEY (ticket_id) REFERENCES ticket (id)
);
''')
cursor.close()  # закрываем курсор
conn.commit()
print('Таблица создана')

conn.close()    # закрываем подключение

# # Добавление таблиц в Postgres
# conn = psycopg2.connect(
#     dbname="postgres",
#     host="localhost",
#     user="postgres",
#     password="postgres",
#     port="5432"
# )
# print('Подключение установлено')
# # conn.autocommit = True  # устанавливаем актокоммит
# cursor = conn.cursor()
#
# cursor.execute('''
# CREATE TABLE embarked (
#     id SERIAL PRIMARY KEY,
#     embarked CHARACTER VARYING(255)
# );
# CREATE TABLE cabin (
#     id SERIAL PRIMARY KEY,
#     cabin CHARACTER VARYING(255)
# );
# CREATE TABLE pclass (
#     id SERIAL PRIMARY KEY,
#     pclass INTEGER,
#     cabin_id INTEGER,
#     FOREIGN KEY (cabin_id) REFERENCES cabin (id)
# );
# CREATE TABLE ticket (
#     id SERIAL PRIMARY KEY,
#     pclass_id INTEGER,
#     cabin_id INTEGER,
#     embarked_id INTEGER,
#     ticket CHARACTER VARYING(255),
#     fare REAL,
#     FOREIGN KEY (pclass_id) REFERENCES pclass (id),
#     FOREIGN KEY (cabin_id) REFERENCES cabin (id),
#     FOREIGN KEY (embarked_id) REFERENCES embarked (id)
# );
# CREATE TABLE passenger (
#     id SERIAL PRIMARY KEY,
#     survived BOOLEAN,
#     name CHARACTER VARYING(255),
#     sex CHARACTER VARYING(255),
#     age REAL,
#     sibSp INTEGER,
#     parch INTEGER,
#     ticket_id INTEGER,
#     FOREIGN KEY (ticket_id) REFERENCES ticket (id)
# );
# ''')
# cursor.commit()
# print('Таблица создана')
#
# cursor.close()  # закрываем курсор
# conn.close()    # закрываем подключение

