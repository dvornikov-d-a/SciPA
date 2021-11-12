from elasticsearch.serializer import JSONSerializer


# Функция отсечения дубликатов публикаций по сравнению их аннотаций,
# а также публикаций без аннотации и списков докладов в рамках конференции
def filter_():
    # Загрузка данных из файла
    with open('../src/0_papers.json', 'r', encoding='utf-8') as f:
        papers = JSONSerializer().loads(f.read())

    # Создание множества уникальных аннотаций
    abstracts_set = set([paper['paperAbstract'] for paper in papers])
    # Подсчёт количества дубликатов:
    doubles_count = len(papers) - len(abstracts_set)

    papers_unique = []
    for paper in papers:
        # Проверка наличия аннотации и что аннотация не представляет собой список докладов
        if paper['paperAbstract'] != '' \
                or paper['paperAbstract'].count('.-') > 10 \
                or paper['paperAbstract'].count('--') > 10:
            continue
        if paper['paperAbstract'] in abstracts_set:
            papers_unique.append(paper)
            abstracts_set.remove(paper['paperAbstract'])

    # Уникальные статьи, имеющие аннотации, сохраняются в файл
    with open('../src/1_papers_unique.json', 'w', encoding='utf-8') as f:
        f.write(JSONSerializer().dumps(papers_unique))
