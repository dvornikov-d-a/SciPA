from frozendict import frozendict

encoding = 'utf-8'

rel_path_0_papers_json = 'src/0_papers.json'
rel_path_1_papers_unique_json = 'src/1_papers_unique.json'
rel_path_2_papers_marked_json = 'src/2_papers_marked.json'
rel_path_3_papers_marked_infinitive_json = 'src/3_papers_marked_infinitive.json'

old_field_id = 'id'
old_field_title = 'title'
old_field_abstract = 'paperAbstract'
old_field_authors = 'authors'
old_field_authors_name = 'name'
old_field_authors_ids = 'ids'
old_field_journal = 'journalName'
old_field_fields = 'fieldOfStudy'
old_field_class = 'match'

old_fields = (old_field_id, old_field_title, old_field_abstract, old_field_authors, old_field_journal, old_field_fields)

field_id_ = 'id_'
field_words_ = 'words_'
field_authors_ = 'authors_'
field_journals_ = 'journals_'
field_fields_ = 'fields_'
field_class_ = 'class_'

classes = (0, 1)

set_names = (field_words_, field_authors_, field_journals_, field_fields_)

data_prev = 'src/papers_'
data_ext = '.csv'

shuffle_count = 5
train_frac = 0.5
k = 5

best_params_ms_json = 'res/best_params_ms.json'
best_params_txt = 'res/best_params.txt'

# Варианты альф
alphas = (1, 0.8, 0.5)

best_alpha = alphas[2]

# Варианты весов
ws = (
    # Слова, авторы, журналы, области
    frozendict(zip(set_names, (1, 1, 1, 1))),
    frozendict(zip(set_names, (1, 4, 3, 2)))
)

best_w = ws[1]

max_vol_frac = 0.8
min_vol = 5

samples_count = 10

vol_ms_json = 'res/vol_ms.json'

threshold_train_size = 50

five_dive_json = 'res/five_dive.json'

