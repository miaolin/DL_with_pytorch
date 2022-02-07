from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import random
import torch
import torch.nn as nn

from letter_utils import n_letters, all_letters, line_to_tensor
from rnn_network import NameClassifyRNN


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


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def random_choice(l):
    return l[random.randint(0, len(l)-1)]


def random_training_example(all_cate, cate_line):
    cate = random_choice(all_cate)
    line = random_choice(cate_line[cate])
    cate_tensor = torch.tensor([all_cate.index(cate)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return cate, line, cate_tensor, line_tensor


def train(model, criterion, lr, cate_tensor, line_tensor):
    hidden = rnn.init_hidden()
    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    loss = criterion(output, cate_tensor)
    loss.backward()

    # manual update the parameters
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-lr)
    return output, loss.item()


if __name__ == "__main__":
    print(unicode_to_ascii('Ślusàrski'))

    # build the category_lines dictioary, a list of names per language
    category_lines = {}
    all_categories = []
    print(find_files('../data/names/*.txt'))
    for filename in find_files('../data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    print(n_categories)
    print(category_lines['Italian'][:5])

    for i in range(10):
        cate, line, cate_tensor, line_tensor = random_training_example(all_categories, category_lines)
        print('category =', cate, '/ line =', line)

    # training the network
    n_hidden = 128
    rnn = NameClassifyRNN(n_letters, n_hidden, 18)
    criterion = nn.NLLLoss()
    learning_rate = 0.005
    n_iters = 1000000
    print_every = 5000
    plot_every = 10000
    cur_loss = 0

    for iter in range(1, n_iters + 1):
        cate, line, cate_tensor, line_tensor = random_training_example(all_categories, category_lines)
        output, loss = train(rnn, criterion, learning_rate, cate_tensor, line_tensor)
        cur_loss += loss

        if iter % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == cate else '✗ (%s)' % cate
            print('%d %d%% %.4f %s / %s %s' % (iter, iter / n_iters * 100, loss, line, guess, correct))
