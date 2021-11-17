from elasticsearch import JSONSerializer
from data_preparing.step_2_marking import old_fields

import nltk
import config as c

from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# Цифры
digits = []
for i in range(10):
    digits.append(str(i))

# Латиница
latins = []
for i in range(ord('z') - ord('a') + 1):
    latins.append(chr(ord('a') + i))

# Знаки препинания и прочие знаки
marks = ['.', ',', '!', '?',
         '-', '–', '—', '_',
         ':', ';',
         '«', '»', '\"',
         '(', ')', '[', ']', '{', '}',
         '|', '#', '№', '$', '%', '^', '&', '*', '+', '=']

# Управляющие символы
manage_symbols = ['\n', '\\', '\t', '\f', '\r', '\v', '\\xa0', '\\u']

# Известные символы
my_keyboard_symbols = digits + latins + marks + ['/', '\\', '\'']


# Вспомогательная функция удаления элементов списка по индексу
def pop_indexes(list_, indexes):
    for k, i in enumerate(indexes):
        list_.pop(i - k)


# 0. Функция приведения к нижнему регистру
def lowercase(string):
    return str.lower(string)


# 1. Функция удаления неизвестных символов
def not_on_board(string):
    new_string = ''
    for c in string:
        if c in my_keyboard_symbols:
            new_string = new_string + c
        else:
            new_string = new_string + ' '
    return new_string


# 2. Функция удаления знаков
def remove_marks(string):
    new_string = ''
    for i, c in enumerate(string):
        if c not in marks + manage_symbols or c == '-' \
                and i != 0 and i != len(string) - 1 and string[i - 1] in latins and string[i + 1] in latins:
            new_string = new_string + c
        else:
            new_string = new_string + ' '
    return new_string


# 3. Функция токенизации
def tokenize(string):
    return [term.strip() for term in string.strip().split(' ')]


# 4. Функция удаления ссылок и e-mail'ов
def no_links_and_emails(word_list):
    indexes = []
    for i, word in enumerate(word_list):
        if word.__contains__('http') or word.__contains__('@'):
            indexes.append(i)
    pop_indexes(word_list, indexes)


# 5. Функция пост-токенизации.
#    (Некоторые слова могут быть разделены слешом, но перед их токенизацией необходимо очистить строку от ссылок.
#     Поэтому процесс назвается посттокенизацией)
def post_tokenize(word_list):
    new_words = []
    indexes = []
    for i, word in enumerate(word_list):
        if word.__contains__('/'):
            indexes.append(i)
            news = word.split('/')
            for new in news:
                new_words.append(new)
    pop_indexes(word_list, indexes)
    word_list = word_list + new_words


# 6. Функция удаления числовых значений
def dig_down(word_list):
    indexes = []
    for i, word in enumerate(word_list):
        if len([char for char in word if char in digits]) > 0:
            indexes.append(i)
    pop_indexes(word_list, indexes)


# 7. Функция удаления стоп-слов
def stop(word_list):
    # Стоп-слова в английском языке
    stop_list = digits + latins + ['about', 'above', 'according', 'across', 'actually', 'ad', 'adj', 'ae', 'af',
                                   'after',
                                   'afterwards', 'ag', 'again', 'against', 'ai', 'al', 'all', 'almost', 'alone',
                                   'along',
                                   'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and',
                                   'another', 'any', 'anyhow', 'anyone', 'anything', 'anywhere', 'ao', 'aq', 'ar',
                                   'are',
                                   'aren', 'aren\'t', 'around', 'arpa', 'as', 'at', 'au', 'aw', 'az', 'author',
                                   'ba', 'bb', 'bd', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been',
                                   'before', 'beforehand', 'begin', 'beginning', 'behind', 'being', 'below', 'beside',
                                   'besides', 'between', 'beyond', 'bf', 'bg', 'bh', 'bi', 'billion', 'bj', 'bm', 'bn',
                                   'bo', 'both', 'br', 'bs', 'bt', 'but', 'buy', 'bv', 'bw', 'by', 'bz',
                                   'ca', 'can', 'can\'t', 'cannot', 'caption', 'cc', 'cd', 'cf', 'cg', 'ch', 'ci', 'ck',
                                   'cl', 'click', 'cm', 'cn', 'co', 'co.', 'com', 'copy', 'could', 'couldn',
                                   'couldn\'t',
                                   'cr', 'cs', 'cu', 'cv', 'cx', 'cy', 'cz', 'copyright',
                                   'de', 'did', 'didn', 'didn\'t', 'dj', 'dk', 'dm', 'do', 'does', 'doesn', 'doesn\'t',
                                   'don', 'don\'t', 'down', 'during', 'dz',
                                   'each', 'ec', 'edu', 'ee', 'eg', 'eh', 'eight', 'eighty', 'either', 'else',
                                   'elsewhere',
                                   'end', 'ending', 'enough', 'er', 'es', 'et', 'etc', 'even', 'ever', 'every',
                                   'everyone',
                                   'everything', 'everywhere', 'except',
                                   'few', 'fi', 'fifty', 'find', 'first', 'five', 'fj', 'fk', 'fm', 'fo', 'for',
                                   'former',
                                   'formerly', 'forty', 'found', 'four', 'fr', 'free', 'from', 'further', 'fx',
                                   'ga', 'gb', 'gd', 'ge', 'get', 'gf', 'gg', 'gh', 'gi', 'gl', 'gm', 'gmt', 'gn', 'go',
                                   'gov', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'gw', 'gy',
                                   'had', 'has', 'hasn', 'hasn\'t', 'have', 'haven', 'haven\'t', 'he', 'he\'d',
                                   'he\'ll',
                                   'he\'s', 'help', 'hence', 'her', 'here', 'here\'s', 'hereafter', 'hereby', 'herein',
                                   'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'hk', 'hm', 'hn', 'home',
                                   'homepage', 'how', 'however', 'hr', 'ht', 'htm', 'html', 'http', 'hu', 'hundred',
                                   'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'i.e.', 'id', 'ie', 'if', 'ii', 'il', 'im', 'in',
                                   'inc', 'inc.', 'indeed', 'information', 'instead', 'int', 'into', 'io', 'iq', 'ir',
                                   'is',
                                   'isn', 'isn\'t', 'it', 'it\'s', 'its', 'itself',
                                   'je', 'jm', 'jo', 'join', 'jp',
                                   'ke', 'kg', 'kh', 'ki', 'km', 'kn', 'koo', 'kp', 'kr', 'kw', 'ky', 'kz', 'keywords',
                                   'la', 'last', 'later', 'latter', 'lb', 'lc', 'least', 'less', 'let', 'let\'s', 'li',
                                   'like', 'likely', 'lk', 'll', 'lr', 'ls', 'lt', 'ltd', 'lu', 'lv', 'ly',
                                   'ma', 'made', 'make', 'makes', 'many', 'maybe', 'mc', 'md', 'me', 'meantime',
                                   'meanwhile', 'mg', 'mh', 'microsoft', 'might', 'mil', 'million', 'miss', 'mk', 'ml',
                                   'mm', 'mn', 'mo', 'more', 'moreover', 'most', 'mostly', 'mp', 'mq', 'mr', 'mrs',
                                   'ms',
                                   'msie', 'mt', 'mu', 'much', 'must', 'mv', 'mw', 'mx', 'my', 'myself', 'mz',
                                   'na', 'namely', 'nc', 'ne', 'neither', 'net', 'netscape', 'never', 'nevertheless',
                                   'new',
                                   'next', 'nf', 'ng', 'ni', 'nine', 'ninety', 'nl', 'no', 'nobody', 'none',
                                   'nonetheless',
                                   'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'np', 'nr', 'nu', 'null', 'nz',
                                   'of', 'off', 'often', 'om', 'on', 'once', 'one', 'one\'s', 'only', 'onto', 'or',
                                   'org',
                                   'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'overall',
                                   'own',
                                   'pa', 'page', 'pe', 'per', 'perhaps', 'pf', 'pg', 'ph', 'pk', 'pl', 'pm', 'pn', 'pr',
                                   'pt', 'pw', 'py',
                                   'qa',
                                   'rather', 're', 'recent', 'recently', 'reserved', 'ring', 'ro', 'ru', 'rw',
                                   'sa', 'same', 'sb', 'sc', 'sd', 'se', 'seem', 'seemed', 'seeming', 'seems', 'seven',
                                   'seventy', 'several', 'sg', 'sh', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should',
                                   'shouldn', 'shouldn\'t', 'si', 'since', 'site', 'six', 'sixty', 'sj', 'sk', 'sl',
                                   'sm',
                                   'sn', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes',
                                   'somewhere', 'sr', 'st', 'still', 'stop', 'su', 'such', 'sv', 'sy', 'sz',
                                   'taking', 'tc', 'td', 'ten', 'text', 'tf', 'tg', 'test', 'th', 'than', 'that',
                                   'that\'ll', 'that\'s', 'the', 'their', 'them', 'themselves', 'then', 'thence',
                                   'there',
                                   'there\'ll', 'there\'s', 'thereafter', 'thereby', 'therefore', 'therein',
                                   'thereupon',
                                   'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'thirty', 'this',
                                   'those', 'though', 'thousand', 'three', 'through', 'throughout', 'thru', 'thus',
                                   'tj',
                                   'tk', 'tm', 'tn', 'to', 'together', 'too', 'toward', 'towards', 'tp', 'tr',
                                   'trillion',
                                   'tt', 'tv', 'tw', 'twenty', 'two', 'tz',
                                   'ua', 'ug', 'uk', 'um', 'under', 'unless', 'unlike', 'unlikely', 'until', 'up',
                                   'upon',
                                   'us', 'use', 'used', 'using', 'uy', 'uz',
                                   'va', 'vc', 've', 'very', 'vg', 'vi', 'via', 'vn', 'vu',
                                   'was', 'wasn', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'web',
                                   'webpage',
                                   'website', 'welcome', 'well', 'were', 'weren', 'weren\'t', 'wf', 'what', 'what\'ll',
                                   'what\'s', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
                                   'whereas',
                                   'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                                   'whither',
                                   'who', 'who\'d', 'who\'ll', 'who\'s', 'whoever', 'whole', 'whom', 'whomever',
                                   'whose',
                                   'why', 'will', 'with', 'within', 'without', 'won', 'won\'t', 'would', 'wouldn',
                                   'wouldn\'t', 'ws', 'www',
                                   'ye', 'yes', 'yet', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your',
                                   'yours',
                                   'yourself', 'yourselves', 'yt', 'yu',
                                   'za', 'zm', 'zr']
    indexes = []
    for i, word in enumerate(word_list):
        if word in stop_list:
            indexes.append(i)
    pop_indexes(word_list, indexes)


# Вспомогательные конструкции для лемматизации
lemmatizer = WordNetLemmatizer()
tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}


def get_wordnet_pos(word_):
    tag = nltk.pos_tag([word_])[0][1][0].upper()
    return tag_dict.get(tag, wordnet.NOUN)


# 8. Функция лемматизации
def lemmatize(word_list):
    for i, word in enumerate(word_list):
        word_list[i] = lemmatizer.lemmatize(word, get_wordnet_pos(word))


# 9. Функция приведения множественного числа существительных к единственному
def single(word_list):
    # Особые формы множественного числа существительных
    special_multi = {'men': 'man', 'women': 'woman', 'children': 'child', 'teeth': 'tooth', 'feet': 'foot',
                     'mice': 'mouse',
                     'geese': 'goose', 'lice': 'louse', 'oxen': 'ox', 'data': 'datum', 'criteria': 'criterion',
                     'geniuses': 'genius', 'formulae': 'formula', 'formulas': 'formula', 'cactuses': 'cactus',
                     'cactii': 'cactus',
                     'memoranda': 'memorandum', 'memorandums': 'memorandum', 'phenomena': 'phenomenon',
                     'bacteria': 'bacterium',
                     'bacterium': 'curriculum', 'analyses': 'analysis', 'bases': 'basis', 'crises': 'crisis',
                     'theses': 'thesis',
                     'hypotheses': 'hypothesis', 'parentheses': 'parenthesis'}
    for i, word in enumerate(word_list):
        if word in special_multi.keys():
            word_list[i] = special_multi[word]
        elif len(word) > 2 and word[-2:] == 'es':
            if word[-3] == 's' or word[-3] == 'z' or word[-3] == 'x' \
                    or len(word) > 3 and (word[-4:-3] == 'ss' or word[-4:-3] == 'sh' or word[-4:-3] == 'ch') \
                    or len(word) > 4 and word[-5:-3] == 'tch':
                word_list[i] = word[:-2]
            elif word[-3] == 'i':
                word_list[i] = f'{word[:-3]}y'
            elif word[-3] == 'v':
                pass
            elif word[-3] == 'o':
                word_list[i] = word[:-2]
        elif len(word) > 1 and word[-1] == 's':
            if word[-2] == 'y':
                word_list[i] = word[:-1]
            elif word[-2] == 'f':
                word_list[i] = word[:-1]
            elif word[-2] == '\'':
                word_list[i] = word[:-2]


# Функция удаления пустых строк
def no_empties(word_list):
    indexes = []
    for i, word in enumerate(word_list):
        if word == '':
            indexes.append(i)
    pop_indexes(word_list, indexes)


def process_text():
    with open(c.rel_path_2_papers_marked_json, 'r', encoding=c.encoding) as f:
        papers = JSONSerializer().loads(f.read())
    for paper in papers:
        for field in [c.old_field_title, c.old_field_title]:
            word_list = tokenize(remove_marks(not_on_board(lowercase(paper[field]))))
            no_links_and_emails(word_list)
            post_tokenize(word_list)
            dig_down(word_list)
            stop(word_list)
            lemmatize(word_list)
            single(word_list)
            no_empties(word_list)
            paper[field] = word_list
        authors = []
        for author in paper[c.old_field_authors]:
            for author_id in author[c.old_field_authors_ids]:
                authors.append(author_id)
        journal = remove_marks(not_on_board(lowercase(paper[c.old_field_journal])))

        paper[c.field_id] = paper[c.old_field_id]
        paper[c.field_words] = paper[c.old_field_title] + paper[c.old_field_abstract]
        paper[c.field_authors] = authors
        paper[c.field_journals] = [journal]
        paper[c.field_fields] = paper[c.old_field_fields]
        paper[c.field_class] = paper[c.old_field_class]

        for old_field in c.old_fields + [c.old_field_class]:
            paper.pop(old_field)
    with open(rel_path_3_papers_marked_infinitive_json, 'w', encoding=c.encoding) as f:
        f.write(JSONSerializer().dumps(papers))
