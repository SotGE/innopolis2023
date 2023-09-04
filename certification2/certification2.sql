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
select
	*
from
	tickets
where
	passenger_name like 'MAKSIM%'
order by
	passenger_name
limit
	10
;

/*
 * 3) Используя SQL язык напишите OLAP запрос к произвольной связке таблиц (в
 * рамках JOIN оператора), используя оператор GROUP BY и любые агрегатные
 * функции count, min, max, sum.
 */
select
	count(aircrafts.aircraft_code),
	sum(aircrafts.range)
from
	aircrafts
full join
	seats
on
	seats.aircraft_code=aircrafts.aircraft_code
group by
	aircrafts.aircraft_code,
	aircrafts.range
order by
	aircrafts.aircraft_code,
	aircrafts.range
;

/*
 * 4) Используя SQL язык примените JOIN операторы (INNER, LEFT, RIGHT) для
 * более чем двух таблиц из модели данных.
 */
select
	*
from
	seats
inner join
	aircrafts
on
	aircrafts.aircraft_code=seats.aircraft_code
left join
	flights
on
	flights.aircraft_code=aircrafts.aircraft_code
right join
	ticket_flights
on
	ticket_flights.flight_id=flights.flight_id
;

/*
 * 5) Создайте виртуальную таблицу VIEW с произвольным именем для SQL запроса
 * из задания 2)
 */
create view air_view as
	select
		*
	from
		tickets
	where
		passenger_name like 'MAKSIM%'
	order by
		passenger_name
	limit
		10
;