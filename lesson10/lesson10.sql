/*
 * -----------------------------------------------------------------------------
 * Лабораторная работа по теме "Язык PL/pgSQL"
 * -----------------------------------------------------------------------------
 */
set search_path to lesson10;

create table student
(
    id serial primary key,
    name text not null,
    total_score integer not null default 0
);
insert into student(id, name, total_score) values (10, 'Иван Иванов', 0);
insert into student(id, name, total_score) values (20, 'Екатерина Андреева', 0);
insert into student(id, name, total_score) values (30, 'Анна Потапова', 0);
insert into student(id, name, total_score) values (40, 'Ильяс Мухаметшин', 0);
insert into student(id, name, total_score) values (50, 'Сергей Петров', 0);

create table activity_scores
(
    id serial primary key,
    student_id integer not null,
    activity_type text not null,
    score integer not null default 0,
    constraint fk_student_id foreign key (student_id) references student(id)
);
insert into activity_scores(student_id, activity_type, score) values (10, 'Queries', 5);
insert into activity_scores(student_id, activity_type, score) values (10, 'Queries', 5);
insert into activity_scores(student_id, activity_type, score) values (20, 'Exam', 45);
insert into activity_scores(student_id, activity_type, score) values (20, 'Certification', 50);
insert into activity_scores(student_id, activity_type, score) values (30, 'Homework', 15);
insert into activity_scores(student_id, activity_type, score) values (30, 'Queries', 5);
insert into activity_scores(student_id, activity_type, score) values (30, 'Homework', 10);
insert into activity_scores(student_id, activity_type, score) values (40, 'Exam', 45);
insert into activity_scores(student_id, activity_type, score) values (40, 'Laboratory', 25);
insert into activity_scores(student_id, activity_type, score) values (40, 'Queries', 5);
insert into activity_scores(student_id, activity_type, score) values (40, 'Homework', 10);
insert into activity_scores(student_id, activity_type, score) values (50, 'Laboratory', 30);
insert into activity_scores(student_id, activity_type, score) values (50, 'Homework', 15);
insert into activity_scores(student_id, activity_type, score) values (50, 'Queries', 5);

/*
 * -----------------------------------------------------------------------------
 * Задача 1: Расчет стипендии для студентов
 * У вас есть две таблицы в базе данных: students и activity_scores. Таблица
 * students содержит информацию о студентах, их идентификаторах и общем
 * балле. Таблица activity_scores содержит информацию о баллах за разные
 * виды деятельности для каждого студента.
 * Создайте таблицу students с колонками:
 * ● id (SERIAL) - идентификатор студента (PRIMARY KEY)
 * ● name (TEXT) - имя студента
 * ● total_score (INTEGER) - общий балл студента
 * Создайте таблицу activity_scores с колонками:
 * ● student_id (INTEGER) - ссылка на студента в таблице students
 * ● activity_type (TEXT) - вид деятельности (например, "Homework",
 * "Exam" и т.д.)
 * ● score (INTEGER) - балл за деятельность
 * При наличии таблицы в базе, можете использовать существующую.
 * Создайте функцию calculate_scholarship, которая будет рассчитывать
 * стипендию для студента. Стипендия зависит от общего балла студента:
 * ● Если общий балл больше или равен 90, стипендия равна 1000.
 * ● Если общий балл больше или равен 80, но меньше 90, стипендия
 * равна 500.
 * ● В остальных случаях, стипендия равна 0.
 * Создайте триггер update_scholarship_trigger, который будет
 * автоматически вызывать функцию calculate_scholarship при
 * обновлении баллов за деятельность в таблице activity_scores.
 * Протестируйте решение, вставив данные о студентах и их баллах за
 * деятельность. Посмотрите, как автоматически обновляется стипендия
 * каждого студента после добавления баллов.
 * -----------------------------------------------------------------------------
 * Решение:
 */
drop function calculate_scholarship cascade;
/* ----------------------------------------------------------------------------- */
create or replace function
	calculate_scholarship()
returns
	trigger as $$
declare
	user_score numeric;
	student_name text;
begin
	-- Получаем балл за деятельность
	select
		sum(score)
	into
		user_score
	from
		activity_scores
	where
		activity_scores.student_id=new.student_id;
	
	-- Имя студента	
	student_name := (
		select
			name
		from
			student
		where
	    	id=new.student_id
    );
	
	-- Условие расчета степендии студента
	if user_score >= 90 then
		raise notice e'Студент: %\nБалл студента: %\nСтепендия: % руб.\n', student_name, user_score, 1000;
	elseif user_score >= 80 and user_score < 90 then
		raise notice e'Студент: %\nБалл студента: %\nСтепендия: % руб.\n', student_name, user_score, 500;
	else
		raise notice e'Студент: %\nБалл студента: %\nСтепендия: % руб.\n', student_name, user_score, 0;
	end if;

	return new;
end;
$$ language plpgsql;
/* ----------------------------------------------------------------------------- */
create or replace trigger
	update_scholarship_trigger
after update on
	activity_scores
for each row
execute function calculate_scholarship();
/* ----------------------------------------------------------------------------- */
update
	activity_scores
set
	score = 50
where
	id = 5
;

/*
 * -----------------------------------------------------------------------------
 * Задача 2: Учет баллов студентов
 * Представьте, что вы разрабатываете систему для учета баллов студентов в
 * университете. Вам необходимо создать функциональность, которая
 * автоматически будет обновлять общий балл каждого студента на основе
 * полученных им баллов за разные виды деятельности.
 * Создайте таблицу students, содержащую следующие поля:
 * ● id (SERIAL, PRIMARY KEY)
 * ● name (TEXT)
 * ● total_score (INTEGER)
 * Создайте таблицу activity_scores, содержащую следующие поля:
 * ● student_id (INTEGER, FOREIGN KEY к полю id таблицы students)
 * ● activity_type (TEXT)
 * ● score (INTEGER)
 * При наличии таблицы в базе, можете использовать существующую.
 * Напишите функцию update_total_score(student_id INTEGER):
 * ● Эта функция должна пересчитывать общий балл студента на
 * основе баллов за разные виды деятельности в таблице
 * activity_scores.
 * ● Используйте цикл для итерации по всем записям в
 * activity_scores для заданного student_id.
 * ● Обновите поле total_score для соответствующего студента в
 * таблице students суммой всех баллов за разные виды
 * деятельности.
 * Напишите триггер, который будет автоматически вызывать функцию
 * update_total_score при вставке новых записей в таблицу
 * activity_scores.
 * Предоставьте примеры использования:
 * ● Вставьте несколько студентов в таблицу students.
 * ● Вставьте записи о баллах за разные виды деятельности в таблицу
 * activity_scores.
 * ● После вставки баллов, убедитесь, что общий балл каждого
 * студента автоматически обновлен в таблице students.
 * -----------------------------------------------------------------------------
 * Решение:
 */
drop function update_total_score cascade;
/* ----------------------------------------------------------------------------- */
create or replace function
	update_total_score()
returns
	trigger as $$
declare
    scores_array integer[];
    i integer;
	scores_sum integer;
	student_name text;
begin
	-- Значение суммы по умолчанию
	scores_sum := 0;

	-- Имя студента	
	student_name := (
		select
			name
		from
			student
		where
	    	id=new.student_id
    );

	-- Массив из баллов за деятельность для студента
	scores_array := array(
		select
			score
		from
			activity_scores
		where
	    	student_id=new.student_id
	);

	-- Перебор массива из баллов за деятельность и сложение в общую сумму
	foreach i in array scores_array loop
    	scores_sum=scores_sum+i;
    end loop;
   
	-- Обновление общего балла у студента
	update
		student
	set
		total_score=scores_sum
	where
		id=new.student_id
	;

	-- Информационный вывод
	raise notice e'Студент: %\nОбщий балл: %\n', student_name, scores_sum;
	return new;
end;
$$ language plpgsql;
/* ----------------------------------------------------------------------------- */
create or replace trigger
	insert_total_score_trigger
after insert on
	activity_scores
for each row
execute function update_total_score();
/* ----------------------------------------------------------------------------- */
insert into activity_scores (
	student_id,
	activity_type,
	score
)
values (
	20,
	'Exam',
	30
);
/* ----------------------------------------------------------------------------- */
select
	*
from
	student
;
/* ----------------------------------------------------------------------------- */
select
	*
from
	activity_scores
;