import config as c
from elasticsearch.serializer import JSONSerializer
from copy import deepcopy
from IPython.display import Markdown, display


# Красивый консольный вывод
def print_mod(string):
    display(Markdown(string))


def mark():
    # Чтение данных
    with open('../src/1_papers_unique.json', 'r', encoding=c.encoding) as f:
        papers_unique = JSONSerializer().loads(f.read())

    # Наполнение списка публикаций, хранящий только поля, используемые для обучения модели
    papers_cut = []
    for unique_paper in papers_unique:
        cut_paper = {}
        for field in c.old_fields:
            cut_paper[field] = deepcopy(unique_paper[field])
        papers_cut.append(cut_paper)

    # Разметка с записью результатов в файл
    with open(c.rel_path_2_papers_marked_json, 'a+', encoding=c.encoding) as f:
        # Костыль: закомментировать в случае продолжения разметки
        f.write('[')
        for i, cut_paper in enumerate(papers_cut, start=1):
            # Костыль: раскомментировать в случае продолжения разметки, вместо _ номер последней публикации
            # if i <= _:
            #     continue
            print_mod(f'**[{i}/{len(papers_cut)}]**')
            print_mod('**Заголовок**')
            print(cut_paper[c.old_field_title])
            print_mod('**Аннотация**')
            print(cut_paper[c.old_field_abstract])
            print_mod('**Авторы**')
            print(*[author[c.old_field_authors_name] for author in cut_paper[c.old_field_authors]], sep=', ', end='.\n')
            print_mod('**Журнал/Конференция**')
            print(cut_paper[c.old_field_journal])
            print_mod('**Предметная область**')
            print(*cut_paper[c.old_field_fields], sep=', ', end='.\n')
            print_mod('**Match...**')
            print('Ваша оценка (1 - подходит, 0 - не подходит): ', end='')
            cut_paper[c.old_field_class] = int(input())
            print('---------------------------------------------------------', end='\n\n')
            if i < len(papers_cut):
                sep = ','
            else:
                sep = ''
            f.write(f'{JSONSerializer().dumps(cut_paper)}{sep}')
        f.write(']')
