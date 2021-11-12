from elasticsearch import JSONSerializer


set_names = ('words_', 'authors_', 'journals_', 'fields_')


def mine_sets_in_list(papers):
    sets = {}
    for s in set_names:
        sets[s] = set()
    for paper in papers:
        for s in set_names:
            set_ = set(paper[s])
            set_.discard('')
            sets[s].update(set_)
    for s in set_names:
        sets[s] = list(sets[s])
    return sets


def write_headers(papers_sets):
    for s in set_names:
        with open(f'src/papers_{s}.csv', 'w', encoding='utf-8') as f:
            head_string = ','.join(['id_'] + papers_sets[s] + ['class_']) + '\n'
            f.write(head_string)


def structurize():
    with open('src/3_papers_marked_infinitive.json', 'r', encoding='utf-8') as f:
        papers = JSONSerializer().loads(f.read())

    paper_sets = mine_sets_in_list(papers)
    write_headers(paper_sets)

    for paper in papers:
        for s in set_names:
            counts = []
            for term in paper_sets[s]:
                counts.append(str(paper[s].count(term)))
            next_list = [paper['id_']] + counts + [str(paper['class_'])]
            next_string = ','.join(next_list) + '\n'
            with open(f'src/papers_{s}.csv', 'a', encoding='utf-8') as f:
                f.write(next_string)



