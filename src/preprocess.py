#!/usr/bin/env python3
# %%
# The goal of this script is to consolidate the following 
# - build_vocab.sh
# - cut_vocab.sh
# - python3 pickle_vocab.py
# - python3 cooc.py
import collections
from subprocess import Popen, PIPE
import re
from scipy.sparse import *    # this script needs scipy >= v0.15
import pickle


DATA_LOCATION="../data/"

## build_vocab.sh
# cat train_pos.txt train_neg.txt |
with open(DATA_LOCATION+"train_pos.txt", "rt") as f_pos:
    with open(DATA_LOCATION+"train_neg.txt", "rt") as f_neg:
        file_content = f_pos.read() + f_neg.read()

# sed "s/ /\n/g" |
file_content = file_content.replace(" ", "\n")


# grep -v "^\s*$" |
file_content_list = [x for x in file_content.split('\n') if not re.match(r"^\s*$", x)]


# sort |
# (Does not produce the same sort order as calling GNU `sort` but the final result in `cooc.pkl` is the same)
file_content_list.sort()

# uniq -c > vocab.txt
unique_words_counted = list(collections.Counter(file_content_list).items())
unique_words_counted

with open(DATA_LOCATION+"my_vocab.txt", "wt") as f:
    f.writelines([f"{count:7} {word}\n" for (word, count) in unique_words_counted])



## cut_vocab.sh

# cat vocab.txt |
with open(DATA_LOCATION+"my_vocab.txt", "rt") as f:
    file_content = f.read()


# sed "s/^\s\+//g" |
file_content = re.sub(r"^\s\+", "subst", file_content, 0)
file_content = '\n'.join([x.lstrip() for x in file_content.split('\n')])


# sort -rn |
# (Currently calls the `sort` binary as its behaviour differs compared to Python sort)
# file_content_list = sorted([x for x in file_content.split('\n')[:-1]], key=lambda x: int(x.split(' ')[0]), reverse=True)
output, error = Popen(
    ['sort', "--numeric-sort", "--reverse"],
    stdout=PIPE, stdin=PIPE, stderr=PIPE
).communicate(file_content.encode('utf-8'))
file_content_list = output.decode('utf-8').split('\n')


# grep -v "^[1234]\s" |
file_content_list = [x for x in file_content_list if not re.match(r"^[1234]\s", x)]
output = '\n'.join(file_content_list).encode('utf-8')

# cut -d' ' -f2
result = (
    '\n'.join([x.split(' ')[1] for x in file_content_list[:-1]])
    +
    '\n'
)

# > vocab_cut.txt
with open(DATA_LOCATION+"my_vocab_cut.txt", "wt") as f:
    f.write(result)



## python3 pickle_vocab.py
vocab = dict()
with open(DATA_LOCATION+'my_vocab_cut.txt') as f:
    for idx, line in enumerate(f):
        vocab[line.strip()] = idx

with open(DATA_LOCATION+'my_vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)



## python3 cooc.py

with open(DATA_LOCATION+'my_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

data, row, col = [], [], []
counter = 1
for fn in [DATA_LOCATION+'train_pos.txt', DATA_LOCATION+'train_neg.txt']:
    with open(fn) as f:
        for line in f:
            tokens = [vocab.get(t, -1) for t in line.strip().split()]
            tokens = [t for t in tokens if t >= 0]
            for t in tokens:
                for t2 in tokens:
                    data.append(1)
                    row.append(t)
                    col.append(t2)

            if counter % 10000 == 0:
                print(counter)
            counter += 1
cooc = coo_matrix((data, (row, col)))
print("summing duplicates (this can take a while)")
cooc.sum_duplicates()
with open(DATA_LOCATION+'my_cooc.pkl', 'wb') as f:
    pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
