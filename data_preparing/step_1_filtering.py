import config as c

from elasticsearch.serializer import JSONSerializer


# Функция отсечения дубликатов публикаций по сравнению их аннотаций,
# а также публикаций без аннотации и списков докладов в рамках конференции
def filter_():
    # Загрузка данных из файла
    with open(c.rel_path_0_papers_json, 'r', encoding=c.encoding) as f:
        papers = JSONSerializer().loads(f.read())

    # Создание множества уникальных аннотаций
    abstracts_set = set([paper[c.old_field_abstract] for paper in papers])
    # Подсчёт количества дубликатов:
    doubles_count = len(papers) - len(abstracts_set)

    papers_unique = []
    for paper in papers:
        # Проверка наличия аннотации и что аннотация не представляет собой список докладов
        if paper[c.old_field_abstract] != '' \
                or paper[c.old_field_abstract].count('.-') > 10 \
                or paper[c.old_field_abstract].count('--') > 10:
            continue
        if paper[c.old_field_abstract] in abstracts_set:
            papers_unique.append(paper)
            abstracts_set.remove(paper[c.old_field_abstract])

    # Уникальные статьи, имеющие аннотации, сохраняются в файл
    with open(c.rel_path_1_papers_unique_json, 'w', encoding=c.encoding) as f:
        f.write(JSONSerializer().dumps(papers_unique))
