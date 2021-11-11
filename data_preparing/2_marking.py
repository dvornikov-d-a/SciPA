from elasticsearch.serializer import JSONSerializer
from copy import deepcopy
from IPython.display import Markdown, display


# Красивый консольный вывод
def print_mod(string):
    display(Markdown(string))


def mark():
    # Чтение данных
    with open('../src/1_papers_unique.json', 'r', encoding='utf-8') as f:
        papers_unique = JSONSerializer().loads(f.read())

    # Наполнение списка публикаций, хранящий только поля, используемые для обучения модели
    papers_cut = []
    for unique_paper in papers_unique:
        cut_paper = {}
        for field in ['id', 'title', 'paperAbstract', 'authors', 'journalName', 'fieldsOfStudy']:
            cut_paper[field] = deepcopy(unique_paper[field])
        papers_cut.append(cut_paper)

    # Разметка с записью результатов в файл
    with open('../src/2_papers_marked.json', 'a+', encoding='utf-8') as f:
        # Костыль: закомментировать в случае продолжения разметки
        f.write('[')
        for i, cut_paper in enumerate(papers_cut, start=1):
            # Костыль: раскомментировать в случае продолжения разметки, вместо _ номер последней публикации
            # if i <= _:
            #     continue
            print_mod(f'**[{i}/{len(papers_cut)}]**')
            print_mod('**Заголовок**')
            print(cut_paper['title'])
            print_mod('**Аннотация**')
            print(cut_paper['paperAbstract'])
            print_mod('**Авторы**')
            print(*[author['name'] for author in cut_paper['authors']], sep=', ', end='.\n')
            print_mod('**Журнал/Конференция**')
            print(cut_paper['journalName'])
            print_mod('**Предметная область**')
            print(*cut_paper['fieldsOfStudy'], sep=', ', end='.\n')
            print_mod('**Match...**')
            print('Ваша оценка (1 - подходит, 0 - не подходит): ', end='')
            cut_paper['match'] = int(input())
            print('---------------------------------------------------------', end='\n\n')
            if i < len(papers_cut):
                sep = ','
            else:
                sep = ''
            f.write(f'{JSONSerializer().dumps(cut_paper)}{sep}')
        f.write(']')
