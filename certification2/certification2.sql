/*
 * -----------------------------------------------------------------------------
 * Лабораторная работа по теме "Промежуточная аттестация 2"
 * -----------------------------------------------------------------------------
 */

/*
 * 0) Скачайте и разверните модель данных на вашем установленной БД PostgreSQL
 * Модель данных развернется в базу данных demo.
 */
-- Запуск CMD:
-- cd C:\Program Files\PostgreSQL\15\bin
-- psql.exe -f "D:\projects\innopolis2023\certification2\air.sql" -U postgres
-- chcp 1251
-- psql -U postgres
-- SET client_encoding="WIN1251";
-- \c demo;
-- SET search_path=bookings, pg_catalog;
-- \dt+

set search_path to bookings;

/*
 * 1) Используя SQL язык и произвольные две таблицы из модели данных
 * необходимо объединить их различными способами (UNION , JOIN)
 */
select
	aircraft_code,
	model as "model",
	null as "fare_conditions"
from
	aircrafts
union
select
	aircraft_code,
	null,
	fare_conditions
from
	seats
group by
	aircraft_code,
	fare_conditions
order by
	aircraft_code
;
/* ----------------------------------------------------------------------------- */
select
	*
from
	aircrafts
full join
	seats
on
	seats.aircraft_code=aircrafts.aircraft_code
order by
	seats.aircraft_code
;

/*
 * 2) Используя SQL язык напишите запрос с любым фильтром WHERE к
 * произвольной таблице и результат отсортируйте (ORDER BY) с ограничением
 * вывода по количеству строк (LIMIT)
 */