import config as c
from elasticsearch import JSONSerializer


def mine_sets_in_list(papers):
    sets = {}
    for s in c.set_names:
        sets[s] = set()
    for paper in papers:
        for s in c.set_names:
            set_ = set(paper[s])
            set_.discard('')
            sets[s].update(set_)
    for s in c.set_names:
        sets[s] = list(sets[s])
    return sets


def write_headers(papers_sets):
    for s in c.set_names:
        with open(f'{c.data_prev}{s}{c.data_ext}', 'w', encoding=c.encoding) as f:
            head_string = ','.join(['id_'] + papers_sets[s] + ['class_']) + '\n'
            f.write(head_string)


def structurize():
    with open(c.rel_path_3_papers_marked_infinitive_json, 'r', encoding=c.encoding) as f:
        papers = JSONSerializer().loads(f.read())

    paper_sets = mine_sets_in_list(papers)
    write_headers(paper_sets)

    for paper in papers:
        for s in c.set_names:
            counts = []
            for term in paper_sets[s]:
                counts.append(str(paper[s].count(term)))
            next_list = [paper[c.field_id_]] + counts + [str(paper[c.field_class_])]
            next_string = ','.join(next_list) + '\n'
            with open(f'{c.data_prev}{s}{c.data_ext}', 'a', encoding=c.encoding) as f:
                f.write(next_string)



