import unicodedata
import re

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def tokenize(sentence, regex=SENTENCE_SPLIT_REGEX, keep=None, remove=None):
    if keep is None:
        keep = ["'s"]
    if remove is None:
        remove = [',', '?']
    sentence = sentence.lower()

    for token in keep:
        sentence = sentence.replace(token, ' ' + token)

    for token in remove:
        sentence = sentence.replace(token, '')

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def _is_punctuation(char):
    if char == '-':
        return False
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def _add_space_around_punc(string):
    pucs = [c for c in string if _is_punctuation(c)]
    for p in pucs:
        p_ = ' ' + p + ' '
        string = string.replace(p, p_)
    return string


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines
