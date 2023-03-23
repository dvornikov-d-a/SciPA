# SciPA
Система рекомендаций научных статей методами машинного обучения

## Содержание

1. [Что это?](#what)
2. [Как это работает?](#how)
3. [Где это использовать?](#where)

### 1. Что это?<a name='what'></a>

Здесь покоится модификация Наивного Байеса для системы рекомендации научных статей. Ключевые особенности:

- весовые коэффициенты для различных групп признаков;
- байесовская модель ранжирования.

То есть помимо слов из текста модель умеет анализировать и такие метаданные, как авторство статьи, источник и область знаний. Также с её помощью можно ранжировать корпус документов.

Модель обучается на оценках пользователя и вырабатывает персональные рекомендации.

### 2. Как это работает?<a name='how'></a>

Исчерпывающая техническая документация :)

![Data Flow](https://github.com/dvornikov-d-a/SciPA/blob/master/Data%20Flow.png?raw=true)

### 3. Где это использовать?<a name='where'></a>

Модель можно адаптировать под любые задачи текстовой классификации и ранжирования. Наработки будут полезны для любых задач NLP и Text Mining.

В */src* находится готовый датасет из числа статей на английском языке, найденным по запросу «Brain Cancer MRI» с фильтром по области знаний «Computer Science» из Semantic Scholar Open Research Corpus (релиз от 01.09.2021). Всего 499 статей.

В `naive_bayes_mod.py` находится сама модель.
