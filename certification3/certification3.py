# %%
!pip install pymorphy3
!pip install transformers
!pip install fse

# %%
import os # работа с папкой и файлами
import re # регулярные выражения, доп вариант к очистке
import string # работа со строкой
import math
import operator
import random
import time
import json
import datetime
import csv
import requests
import joblib

import pandas as pd
import numpy as np

from wordcloud import WordCloud, ImageColorGenerator # визуальное отображение
import pymorphy3 # работа с русским языком, pymorphy3

import matplotlib.pyplot as plt # визуальное отображение
import seaborn as sns
import plotly.express as px

from tqdm.auto import tqdm  # индикатор выполнения

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split # Деление выборки на тестовые и тренировочные данные
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# python -m pip install transformers
from transformers import BertTokenizer # токенизатор BERT
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertTokenizerFast
from transformers import BertForQuestionAnswering
from transformers import DistilBertForQuestionAnswering

import nltk # работа с пакетами языков
from nltk import word_tokenize, ngrams # токенизация и деление на n граммы
from nltk.corpus import stopwords # стопслова, extend

# python -m pip install tensorflow
# python -m pip install torch torchvision torchaudio
# import tensorflow as tf
# from tensorflow import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score # Критерий качества, точности

import gensim
import gensim.downloader as api

# Для установки "pip install fse" необходимо:
# Python 3.10.13
# Visual Studio Build Tools 2022 (Включено: создание современных приложений C++ для Windows с помощью средств, включая MSVC, Clang, CMake или MSBuild)
from fse import SplitIndexedList
from fse.models import uSIF

# скачаем словарь стоп-слов
nltk.download("stopwords")

# скачиваем модель, которая будет делить на предложения
nltk.download('punkt')

# %%
# Парсер файлов .json в DataFrame
def convert(f, path=['data', 'paragraphs', 'qas', 'answers']):
  file = json.loads(open(f).read())
  # разбор различных уровней в json-файле
  js = pd.json_normalize(file , path)
  m = pd.json_normalize(file, path[:-1])
  r = pd.json_normalize(file, path[:-2])
  # объединение их в один датафрейм
  m['context'] = np.repeat(r['context'].values, r.qas.str.len())
  js['q_idx'] = np.repeat(m['id'].values,m['answers'].str.len())
  data = m[['id', 'question', 'context', 'answers']].set_index('id').reset_index()
  # data = pd.concat([m.reset_index()[['id', 'question', 'context']].set_index('id'), js.reset_index().set_index('q_idx')], axis=1, sort=False).reset_index()
  data['context_id'] = data['context'].factorize()[0]
  return data

# Просмотр файлов в директории
print(os.listdir("archive"))

# %%
# Тренировочные данные
train = convert('archive/train-v1.1.json')
train

# %%
# Валидационные данные
valid = convert('archive/dev-v1.1.json')
valid

# %%
# Проверка на заполненность вопросов тренировочных данных
for i, value in enumerate(train['answers']):
  data = json.loads(json.dumps(value[0]))
  if 'answer_start' not in data:
    print(value)
  if 'text' not in data:
    print(value)

# %%
# Проверка на заполненность вопросов валидационных данных
for i, value in enumerate(valid['answers']):
  data = json.loads(json.dumps(value[0]))
  if 'answer_start' not in data:
    print(value)
  if 'text' not in data:
    print(value)

# %%
# Размер тренировочных данных
print(f'Размер тренировочных данных: {train.shape}')

# Количество уникальных контекстов в наборе тренировочных данных
print(f'Количество уникальных контекстов в наборе тренировочных данных: {train["context_id"].unique().size}')

# Список уникальных контекстов в наборе тренировочных данных
train_unique = train[['context', 'context_id']].drop_duplicates().reset_index(drop=True)
train_unique

# %%
# Размер валидационных данных
print(f'Размер валидационных данных: {valid.shape}')

# Количество уникальных контекстов в наборе валидационных данных
print(f'Количество уникальных контекстов в наборе валидационных данных: {valid["context_id"].unique().size}')

# Список уникальных контекстов в наборе валидационных данных
valid_unique = valid[['context', 'context_id']].drop_duplicates().reset_index(drop=True)
valid_unique

# %%
# Контексты тренировочных данных
train_contexts = train['context'].to_numpy().tolist()
train_contexts

# %%
# Контексты валидационных данных
valid_contexts = valid['context'].to_numpy().tolist()
valid_contexts

# %%
# Вопросы тренировочных данных
train_questions = train['question'].to_numpy().tolist()
train_questions

# %%
# Вопросы валидационных данных
valid_questions = valid['question'].to_numpy().tolist()
valid_questions

# %%
# Ответы тренировочных данных
train_answers = []
for i, value in enumerate(train['answers'].to_numpy().tolist()):
  train_answers.append(value[0]['text'])
train_answers

# %%
# Ответы валидационных данных
valid_answers = []
for i, value in enumerate(valid['answers'].to_numpy().tolist()):
  valid_answers.append(value[0]['text'])
valid_answers

# %%
# Общие тренировочные данные
train_merge = pd.merge(pd.DataFrame(train_contexts, columns=['contexts']), pd.DataFrame(train_questions, columns=['questions']), left_index=True, right_index=True)
train_data = pd.merge(train_merge, pd.DataFrame(train_answers, columns=['answers']), left_index=True, right_index=True)
train_data

# %%
# Общие валидационные данные
valid_merge = pd.merge(pd.DataFrame(valid_contexts, columns=['contexts']), pd.DataFrame(valid_questions, columns=['questions']), left_index=True, right_index=True)
valid_data = pd.merge(valid_merge, pd.DataFrame(valid_answers, columns=['answers']), left_index=True, right_index=True)
valid_data

# %%
# Исправление символов конечного положения в тренировочных данных
for answer, text in zip(train['answers'].values.tolist(), train['context'].values.tolist()):
  real_answer = answer[0]['text']
  answer_start_index = answer[0]['answer_start']
  answer_end_index = answer_start_index + len(real_answer)
  if text[answer_start_index : answer_end_index] == real_answer:
    answer[0]['answer_end'] = answer_end_index
  elif text[answer_start_index - 1 : answer_end_index - 1] == real_answer:
    answer[0]['answer_start'] = answer_start_index - 1
    answer[0]['answer_end'] = answer_end_index - 1
  elif text[answer_start_index - 2 : answer_end_index - 2] == real_answer:
    answer[0]['answer_start'] = answer_start_index - 2
    answer[0]['answer_end'] = answer_end_index - 2

train_answers_positions = []
for i, value in enumerate(train['answers'].values.tolist()):
  temp = []
  for j, item in enumerate(value):
    if 'answer_end' in item:
      temp.append(item['answer_start'])
      temp.append(item['text'])
      temp.append(item['answer_end'])
  train_answers_positions.append(temp)

pd.DataFrame(train_answers_positions, columns=['answer_start', 'text', 'answer_end'])

# %%
# Исправление символов конечного положения в валидационных данных
for answer, text in zip(valid['answers'].values.tolist(), valid['context'].values.tolist()):
  real_answer = answer[0]['text']
  answer_start_index = answer[0]['answer_start']
  answer_end_index = answer_start_index + len(real_answer)
  if text[answer_start_index : answer_end_index] == real_answer:
    answer[0]['answer_end'] = answer_end_index
  elif text[answer_start_index - 1 : answer_end_index - 1] == real_answer:
    answer[0]['answer_start'] = answer_start_index - 1
    answer[0]['answer_end'] = answer_end_index - 1
  elif text[answer_start_index - 2 : answer_end_index - 2] == real_answer:
    answer[0]['answer_start'] = answer_start_index - 2
    answer[0]['answer_end'] = answer_end_index - 2

valid_answers_positions = []
for i, value in enumerate(valid['answers'].values.tolist()):
  temp = []
  for j, item in enumerate(value):
    if 'answer_end' in item:
      temp.append(item['answer_start'])
      temp.append(item['text'])
      temp.append(item['answer_end'])
  valid_answers_positions.append(temp)

pd.DataFrame(valid_answers_positions, columns=['answer_start', 'text', 'answer_end'])

# %%
# Общие тренировочные данные с ответами
train_data_answers = pd.merge(train, pd.DataFrame(train_answers_positions, columns=['answer_start', 'text', 'answer_end']), left_index=True, right_index=True)
train_data_answers

# %%
# Общие валидационные данные с ответами
valid_data_answers = pd.merge(valid, pd.DataFrame(valid_answers_positions, columns=['answer_start', 'text', 'answer_end']), left_index=True, right_index=True)
valid_data_answers

# %%
# Очистка тескта
def clean_text(text):
  text = str(text)
  text = text.lower()
  words = re.sub(r'[^\w\s\.\?]', '', text).split()
  return " ".join([word for word in words])

train_data_answers['question'] = train_data_answers['question'].apply(lambda x: clean_text(x))
train_data_answers['context'] = train_data_answers['context'].apply(lambda x: clean_text(x))
train_data_answers['text'] = train_data_answers['text'].apply(lambda x: clean_text(x))

train_all_question = '. '.join(list(train_data_answers['question']))
train_all_answer = '. '.join(list(train_data_answers['text']))
train_all_context = '. '.join(list(train_data_answers['context']))
train_text = train_all_question + train_all_answer + train_all_context
train_split = SplitIndexedList(train_text.split('.'))

valid_data_answers['question'] = valid_data_answers['question'].apply(lambda x: clean_text(x))
valid_data_answers['context'] = valid_data_answers['context'].apply(lambda x: clean_text(x))
valid_data_answers['text'] = valid_data_answers['text'].apply(lambda x: clean_text(x))

valid_all_question = '. '.join(list(valid_data_answers['question']))
valid_all_answer = '. '.join(list(valid_data_answers['text']))
valid_all_context = '. '.join(list(valid_data_answers['context']))
valid_text = valid_all_question + valid_all_answer + valid_all_context
valid_split = SplitIndexedList(valid_text.split('.'))

# %%
# Разделение на слова, преобразование для обучения тренировочных данных
train_dataset = []
train_title = ""

for i in range(0, len(train_data_answers), 2):
  this_title = train_data_answers.iloc[i]['id']
  if (this_title != train_title):
    train_title = this_title
    text = train_data_answers.iloc[i]['context']
    splitted = text.split(sep='.')
    for j in range(len(splitted)):
      text = splitted[j]
      if(text!=''):
        words = text.split()
        train_dataset.append(words)
  train_dataset.append(train_data_answers.iloc[i]['question'].split())
  train_dataset.append(train_data_answers.iloc[i]['text'].split())
train_dataset

# %%
# Разделение на слова, преобразование для обучения валидационных данных
valid_dataset = []
valid_title = ""

for i in range(0, len(valid_data_answers), 2):
  this_title = valid_data_answers.iloc[i]['id']
  if (this_title != valid_title):
    valid_title = this_title
    text = valid_data_answers.iloc[i]['context']
    splitted = text.split(sep='.')
    for j in range(len(splitted)):
      text = splitted[j]
      if(text!=''):
        words = text.split()
        valid_dataset.append(words)
  valid_dataset.append(valid_data_answers.iloc[i]['question'].split())
  valid_dataset.append(valid_data_answers.iloc[i]['text'].split())
valid_dataset

# %%
# Word2Vec и обучение модели для тренировочных данных
train_model_wv = gensim.models.Word2Vec(train_dataset, vector_size=100, window=8, min_count=1, sg=0, workers=8)
train_model_wv.train(train_dataset, total_examples=len(train_dataset), compute_loss=True, epochs=50)

# %%
# Word2Vec и обучение модели для валидационных данных
valid_model_wv = gensim.models.Word2Vec(valid_dataset, vector_size=100, window=8, min_count=1, sg=0, workers=8)
valid_model_wv.train(valid_dataset, total_examples=len(valid_dataset), compute_loss=True, epochs=50)

# %%
def get_embedding(model_wv, sentence):
  pos_sum = [0.0 for i in range(100)]
  num = 0
  words = sentence.split()
  for i in words:
    try:
      embed = model_wv.wv[i]
    except:
      continue
    else:
      pos_sum += embed
      num +=1
  if(num==0):
    return pos_sum
  else:
    pos_sum /= num
    return pos_sum

# Предсказание модели с использованием евклидова расстояния
def get_answer(model_wv, question, answer_para):
  question_embedding = get_embedding(model_wv, rem_stop(question))
  min_distance = math.inf
  answer = 0
  for i in range(len(answer_para)):
    answer_embedding = get_embedding(model_wv, rem_stop(answer_para[i]))
    distance = np.linalg.norm(question_embedding-answer_embedding)
    if (distance < min_distance):
      answer = i
      # print(answer)
      min_distance = distance
  return answer_para[answer]

# Без стоп слов
def rem_stop(sentence):
  strr=''
  my_string = sentence.split()
  for i in range(len(my_string)):
    if my_string[i] not in stopwords.words('english'):
      strr = strr+' '+my_string[i]
  return strr[1:]

# Предсказание модели с использованием косинусного сходства
def get_answer_cosine(model_wv, question, answer_para):
  question_embedding = get_embedding(model_wv, rem_stop(question))
  max_similarity = -math.inf
  answer = 0
  for i in range(len(answer_para)):
    answer_embedding = get_embedding(model_wv, rem_stop(answer_para[i]))
    similarity = cosine_similarity(np.expand_dims(question_embedding,0), np.expand_dims(answer_embedding,0))
    if (similarity > max_similarity):
      answer = i
      max_similarity = similarity
  return answer_para[answer]

# Разбиваем текст статьи на отдельные предложения
def contextToSents(my_text):
  temp_sentences = my_text.split(sep='.')
  sentences=[]
  for i in range(len(temp_sentences)):
    if(temp_sentences[i]!=''):
      sentences.append(temp_sentences[i])
  return sentences

# %%
# Вопрос из тренировочных данных
index = 296
my_text = train_data_answers.iloc[index]['context']
temp_sentences = my_text.split(sep='.')
train_sentences = []

for i in range(len(temp_sentences)):
  if(temp_sentences[i]!=''):
    train_sentences.append(temp_sentences[i])

train_my_question = train_data_answers.iloc[index]['question']
train_my_question

# %%
# Вопрос из валидационных данных
index = 296
my_text = valid_data_answers.iloc[index]['context']
temp_sentences = my_text.split(sep='.')
valid_sentences = []

for i in range(len(temp_sentences)):
  if(temp_sentences[i]!=''):
    valid_sentences.append(temp_sentences[i])

valid_my_question = valid_data_answers.iloc[index]['question']
valid_my_question

# %%
# Word2Vec outputs из тренировочных данных
print(f'Вопрос:\n{train_my_question}')
print(f'\nВопрос без стоп слов:\n{rem_stop(train_my_question)}')
print(f'\nОтвет:\n{train_data_answers.iloc[index]["text"]}')
print(f'\nПредсказание модели с использованием евклидова расстояния:\n{get_answer(train_model_wv, train_my_question, train_sentences)}')
print(f'\nПредсказание модели с использованием косинусного сходства:\n{get_answer_cosine(train_model_wv, train_my_question, train_sentences)}')

# %%
# Word2Vec outputs из валидационных данных
print(f'Вопрос:\n{valid_my_question}')
print(f'\nВопрос без стоп слов:\n{rem_stop(valid_my_question)}')
print(f'\nОтвет:\n{valid_data_answers.iloc[index]["text"]}')
print(f'\nПредсказание модели с использованием евклидова расстояния:\n{get_answer(valid_model_wv, valid_my_question, valid_sentences)}')
print(f'\nПредсказание модели с использованием косинусного сходства:\n{get_answer_cosine(valid_model_wv, valid_my_question, valid_sentences)}')

# %%
# Загрузка предтренированной модели BERT
# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# %%
# Загрузка предтренированного токенайзера BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# %%
# Базовая конфигурация предтренированного токенайзера BERT
model.base_model.config

# %%
pytorch_total_params = sum(p.numel() for p in model.base_model.parameters() if p.requires_grad)
pytorch_trainable_params = sum(p.numel() for p in model.base_model.parameters() )
print("Total number of params", pytorch_total_params)
print("Total number of trainable params", pytorch_trainable_params)

# %%
# Токенайзер тренировочных данных
train_encodings = tokenizer(
  train_data['contexts'].values.tolist(),
  train_data['questions'].values.tolist(),
  truncation=True,
  padding=True,
  max_length=512,
  return_tensors='pt'
)
train_encodings

# %%
# Токенайзер валидационных данных
validation_encodings = tokenizer(
  valid_data['contexts'].values.tolist(),
  valid_data['questions'].values.tolist(),
  truncation=True,
  padding=True,
  max_length=512,
  return_tensors='pt'
)
validation_encodings

# %%
def get_top_answers(possible_starts, possible_ends, input_ids):
  answers = []
  for start,end in zip(possible_starts, possible_ends):
    #+1 for end
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end+1]))
    answers.append( answer )
  return answers

def answer_question(question, context, topN):
  inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
  input_ids = inputs["input_ids"].tolist()[0]

  text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
  model_out = model(**inputs)

  answer_start_scores = model_out["start_logits"]
  answer_end_scores = model_out["end_logits"]

  possible_starts = np.argsort(answer_start_scores.cpu().detach().numpy()).flatten()[::-1][:topN]
  possible_ends = np.argsort(answer_end_scores.cpu().detach().numpy()).flatten()[::-1][:topN]

  #get best answer
  answer_start = torch.argmax(answer_start_scores)
  answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
  answers = get_top_answers(possible_starts, possible_ends, input_ids)

  return {
    "answer":answer,
    "answer_start":answer_start,
    "answer_end":answer_end,
    "input_ids":input_ids,
    "answer_start_scores":answer_start_scores,
    "answer_end_scores":answer_end_scores,
    "inputs":inputs,
    "answers":answers,
    "possible_starts":possible_starts,
    "possible_ends":possible_ends
  }

# %%
# Контекст
text = r"""
kathmandu metropolitan city kmc in order to promote international relations has established an international relations secretariat irc.
kmcs first international relationship was established in 1975 with the city of eugene oregon united states.
this activity has been further enhanced by establishing formal relationships with 8 other cities motsumoto city of japan rochester of the usa yangon formerly
rangoon of myanmar xian of the peoples republic of china minsk of belarus and pyongyang of the democratic republic of korea.
kmcs constant endeavor is to enhance its interaction with saarc countries other international agencies and many other major cities of the world to achieve better urban management and developmental programs for kathmandu.
"""

# Список вопросов
questions = [
  "what is kmc an initialism of?",
  "in what year did kathmandu create its initial international relationship?",
  "with what belorussian city does kathmandu have a relationship?"
]

# Список ответов
for q in questions:
  answer_map = answer_question(q, text, 5)
  print("Question:", q)
  print("Answers:")
  [print((index+1)," ) ",ans) for index, ans in enumerate(answer_map["answers"]) if len(ans) > 0]

# %%
# Выгружаем модель
torch.save(model, f"model.pth")

# %%
# Загружаем натренированную модель
bert = torch.load(f"model.pth")
bert.eval()


