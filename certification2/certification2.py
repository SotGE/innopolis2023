import psycopg2
from psycopg2 import Error, sql

base='demo'
user='postgres'
password='postgres'
host='127.0.0.1'
port='5432'

try:
  connection = psycopg2.connect(
    dbname=base,
    user=user,
    password=password,
    host=host,
    port=port,
    options="-c search_path=bookings"
  )
  cursor = connection.cursor()

  print("Информация по подключению:")
  print(connection.get_dsn_parameters())

  cursor.execute("select * from air_view")
  fetch = cursor.fetchall()
  print(fetch)

  cursor.close()
  connection.close()
except(Exception, Error) as error:
  print("Возникло исключение при работе с Postgres: ", error)