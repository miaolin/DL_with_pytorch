from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata

from letter_utils import all_letters


def find_files(path):
    return glob.glob(path)


# turn a unicode string to plain ASCII,
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )


def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


if __name__ == "__main__":
    print(unicode_to_ascii('Ślusàrski'))

    # build the category_lines dictioary, a list of names per language
    category_lines = {}
    all_categories = []
    print(find_files('data/names/*.txt'))
    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    print(n_categories)
    print(category_lines['Italian'][:5])

