/*
 * a. Напишите SQL запрос который возвращает имена студентов и их аккаунт
 * в Telegram у которых родной город “Казань” или “Москва”. Результат
 * отсортируйте по имени студента в убывающем порядке
 */
select
	name, telegram_contact
from
	student
where
	city='Казань' or city='Москва'
order by
	name desc
;



/*
 * b. Напишите SQL запрос который возвращает данные по университетам в
 * следующем виде (один столбец со всеми данными внутри) с сортировкой
 * по полю “полная информация”
 * ------------------------------------------------------
 * "полная информация"
 * ------------------------------------------------------
 * университет: Иннополис; количество студентов: 1077
 * университет: КФУ; количество студентов: 50000
 * университет: МГУ; количество студентов: 38000
 * университет: МФТИ; количество студентов: 7000
 * университет: Сколково; количество студентов: 1070
 * ------------------------------------------------------
 */
select
	format('университет: %s; количество студентов: %s', college.name, college.size) as "полная информация"
from
	college
order by
	"полная информация" asc
;



/*
 * c. Напишите SQL запрос который возвращает список университетов и
 * количество студентов, если идентификатор университета должен быть
 * выбран из списка 10, 30, 50. Пожалуйста примените конструкцию IN.
 * Результат запроса отсортируйте по количеству студентов И затем по
 * наименованию университета.
 */
select
	name, size
from
	college
where
	id in (10, 30, 50)
order by
	size asc,
	name asc
;



/*
 * d. Напишите SQL запрос который возвращает список университетов и
 * количество студентов, если идентификатор университета НЕ должен
 * соответствовать значениям из списка 10, 30, 50. Пожалуйста в основе
 * примените конструкцию IN. Результат запроса отсортируйте по
 * количеству студентов И затем по наименованию университета.
 */
select
	name, size
from
	college
where
	id not in (10, 30, 50)
order by
	size asc,
	name asc
;



/*
 * e. Напишите SQL запрос который возвращает название online курсов
 * университетов и количество заявленных слушателей. Количество
 * заявленных слушателей на курсе должно быть в диапазоне от 27 до 310
 * студентов. Результат отсортируйте по названию курса и по количеству
 * заявленных слушателей в убывающем порядке для двух полей.
 */
select
	name as "название online курсов университетов",
	amount_of_students as "количество заявленных слушателей"
from
	course
where
	is_online=true
	and amount_of_students between 27 and 310
order by
	name desc,
	amount_of_students desc
;



/*
 * f. Напишите SQL запрос который возвращает имена студентов и название
 * курсов университетов в одном списке. Результат отсортируйте в
 * убывающем порядке. Пример части результата представлен ниже
 * ------------------------------------------------------
 * name
 * ------------------------------------------------------
 * Цифровая трансформация
 * Сергей Петров
 * ------------------------------------------------------
 */
select
	name
from
	student
union
select
	name
from
	course
order by
	name desc
;



/*
 * g. Напишите SQL запрос который возвращает имена университетов и
 * название курсов в одном списке, но с типом что запись является или
 * “университет” или “курс”. Результат отсортируйте в убывающем порядке
 * по типу записи и потом по имени. Пример части результата представлен
 * ниже
 * ------------------------------------------------------
 * name        | object_type
 * ------------------------------------------------------
 * Иннополис   | университет
 * КФУ         | университет
 * …           | …
 * Data Mining | курс
 * …           | …
 * ------------------------------------------------------
 */
select
	name,
	'университет' as object_type
from
	college
union
select
	name,
	'курс' as object_type
from
	course
order by
	object_type desc,
	name
;



/*
 * h. Напишите SQL запрос который возвращает название курса и количество
 * заявленных студентов в отсортированном списке по количеству
 * слушателей в возрастающем порядке, НО запись с количеством
 * слушателей равным 300 должна быть на первом месте. Ограничьте
 * вывод данных до 3 строк. Пример результата представлен ниже
 * ------------------------------------------------------
 * name                 | amount_of_students
 * ------------------------------------------------------
 * Введение в РСУБД     | 300
 * Data Mining          | 10
 * Актерское мастерство | 15
 * ------------------------------------------------------
 * Подсказка: используйте в ORDER BY синтаксический
 * элемент CASE … END.
 */
select
	name,
	amount_of_students
from
	course
order by
	(case when amount_of_students=300 then course end),
	amount_of_students
limit
	3
;



/*
 * i. Напишите DML запрос который создает новый offline курс со
 * следующими характеристиками:
 * - id = 60
 * - название курса = Machine Learning
 * - количество студентов = 17
 * - курс проводится в том же университете что и курс Data Mining
 * Предоставьте INSERT выражение которое заполняет необходимую
 * таблицу данными
 * Приложите скрин результата запроса к данным курсов после
 * выполнения команды INSERT к таблице которая была изменена.
 */
select
	*
from
	course
;
------------------------------------------------------
with
	search_course
as (
    select
    	id as search_course_id
    from
    	course
    where
    	name='Data Mining'
)
insert into course
	(id, name, is_online, amount_of_students, college_id)
select
	60, 'Machine Learning', false, 17, search_course_id
from
	search_course
;
------------------------------------------------------
select
	*
from
	course
;



/*
 * j. Напишите SQL скрипт который подсчитывает симметрическую разницу
 * множеств A и B.
 * (A \ B) ⋃ (B \ A)
 * где A - таблица course, B - таблица student_on_course, “\” - это разница
 * множеств, “⋃” - объединение множеств. Необходимо подсчитать на
 * основании атрибута id из обеих таблиц. Результат отсортируйте по 1
 * столбцу. Пример результата представлен ниже.
 * ------------------------------------------------------
 * id
 * ------------------------------------------------------
 * 70
 * 80
 * 90
 * 100
 * …
 * ------------------------------------------------------
 */
(
	(
		select
			id
		from
			course
	)
	except
	(
		select
			id
		from
			student_on_course
	)
)
union
(
	(
		select
			id
		from
			student_on_course
	)
	except
	(
		select
			id
		from
			course
	)
)
order by
	id asc
;



/*
 * k. Напишите SQL запрос который вернет имена студентов, курс на котором
 * они учатся, названия их родных университетов (в которых они
 * официально учатся) и соответствующий рейтинг по курсу. С условием
 * что рассматриваемый рейтинг студента должен быть строго больше (>)
 * 50 баллов и размер соответствующего ВУЗа должен быть строго больше
 * (>) 5000 студентов. Результат необходимо отсортировать по первым двум
 * столбцам. Обратите внимание на часть ответа ниже с учетом
 * именования выходных атрибутов вашего запроса
 * -----------------------------------------------------------------------------
 * student_name       | course_name          | student_college | student_rating
 * -----------------------------------------------------------------------------
 * Анна Потапова      | Нейронные сети       | МФТИ            | 76
 * Екатерина Андреева | Актерское мастерство | МГУ             | 95
 * …                  | …                    | …               | …
 * -----------------------------------------------------------------------------
 */
select
	student.name as "student_name",
	course.name as "course_name",
	college.name as "student_college",
	student_on_course.student_rating as "student_rating"
from
	student
left join
	student_on_course
on
	student_on_course.student_id=student.id
left join
	course
on
	course.id=student_on_course.course_id
left join
	college
on
	college.id=student.college_id
where
	student_on_course.student_rating > 50
	and college.size > 5000
order by
	"student_name" asc,
	"course_name" asc
;



/*
 * l. Выведите уникальные семантические пары студентов, родной город
 * которых один и тот же. Результат необходимо отсортировать по первому
 * столбцу. Семантически эквивалентная пара является пара студентов
 * например (Иванов, Петров) = (Петров, Иванов), в этом случае должна
 * быть выведена одна из пар. Обратите внимание на ответ ниже с учетом
 * именования выходных атрибутов вашего запроса
 * ------------------------------------------------------
 * student_1        | student_2          | city
 * ------------------------------------------------------
 * Ильяс Мухаметшин | Иван Иванов        | Казань
 * Сергей Петров    | Екатерина Андреева | Москва
 * ------------------------------------------------------
 */
select distinct on ("city")
	*
from
(
	select
		student_1.name as "student_1"
		, student_2.name as "student_2"
		, student_2.city as "city"
	from
		student as student_1
	left join
		student as student_2
	on
		student_2.city=student_1.city
		and not student_2.name=student_1.name
) as list
where
	not "city"='NULL'
group by
	"city",
	"student_1",
	"student_2"
order by
	"city" asc,
	"student_1" desc
;



/*
 * m. Напишите SQL запрос который возвращает количество студентов,
 * сгруппированных по их оценке. Результат отсортируйте по названию
 * оценки студента. Формула выставления оценки представлена ниже как
 * псевдокод.
 * 
 * ЕСЛИ оценка < 30 ТОГДА неудовлетворительно
 * ЕСЛИ оценка >= 30 И оценка < 60 ТОГДА удовлетворительно
 * ЕСЛИ оценка >= 60 И оценка < 85 ТОГДА хорошо
 * В ОСТАЛЬНЫХ СЛУЧАЯХ отлично
 * 
 * Пример результата ниже. Обратите внимание на именование
 * результирующих столбцов в вашем решении. Курс “Machine Learning”, так
 * как у него нет студентов - проигнорируйте, используя соответствующий
 * тип JOIN.
 * ------------------------------------------------------
 * оценка              | количество студентов
 * ------------------------------------------------------
 * неудовлетворительно | 2
 * отлично             | 3
 * удовлетворительно   | 3
 * хорошо              | 5
 * ------------------------------------------------------
 */
select
	list."student_rating" as "оценка"
	, count(student.name) as "количество студентов"
from
(
	select
		case
			when
				student_on_course.student_rating < 30
			then
				'неудовлетворительно'
			when
				student_on_course.student_rating >= 30
				and student_on_course.student_rating < 60
			then
				'удовлетворительно'
			when
				student_on_course.student_rating >= 60
				and student_on_course.student_rating < 85
			then
				'хорошо'
			else
				'отлично'
		end as "student_rating"
		, student_on_course.student_id as "student_id"
	from
		student_on_course
) as list
left join
	student
on
	student.id="student_id"
group by
	list."student_rating"
order by
	list."student_rating"
;



/*
 * n. Дополните SQL запрос из задания a), с указанием вывода имени курса и
 * количество оценок внутри курса. Результат отсортируйте по названию
 * курса и оценки студента. Пример части результата ниже.
 * Обратите внимание на именование результирующих столбцов в вашем
 * решении. Курс “Machine Learning”, так как у него нет студентов -
 * проигнорируйте, используя соответствующий тип JOIN.
 * -----------------------------------------------------------------------------
 * курс                   | оценка              | количество студентов
 * -----------------------------------------------------------------------------
 * Data Mining            | неудовлетворительно | 1
 * Data Mining            | хорошо              | 2
 * Актерское мастерство   | отлично             | 2
 * …                      | …                   | …
 * Цифровая трансформация | удовлетворительно   | 2
 * -----------------------------------------------------------------------------
 */
select
	course.name as "курс"
	, list."student_rating" as "оценка"
	, count(student.name) as "количество студентов"
from
(
	select
		case
			when
				student_on_course.student_rating < 30
			then
				'неудовлетворительно'
			when
				student_on_course.student_rating >= 30
				and student_on_course.student_rating < 60
			then
				'удовлетворительно'
			when
				student_on_course.student_rating >= 60
				and student_on_course.student_rating < 85
			then
				'хорошо'
			else
				'отлично'
		end as "student_rating"
		, student_on_course.student_id as "student_id"
		, student_on_course.student_rating as "student_rating_number"
	from
		student_on_course
) as list
left join
	student
on
	student.id=list."student_id"
left join
	student_on_course
on
	student_on_course.student_id=student.id
left join
	course
on
	course.id=student_on_course.course_id
group by
	"курс"
	, "оценка"
order by
	"курс" asc
;