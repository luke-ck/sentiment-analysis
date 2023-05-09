#/usr/bin/env python3
# %%

import json
from tqdm import tqdm

# I tried symspell but it seemed to ignore the max_edit distance
# which gave me weird results like `man*hat*ten` -> `dlck`
from spellchecker import SpellChecker


# wget https://www.freewebheaders.com/download/files/facebook-bad-words-list_comma-separated-text-file_2022_05_01.zip
# wget https://www.freewebheaders.com/download/files/youtube-blacklist-words_comma-separated-text-file.zip

# We only need the full dataset as the rest are subsets
# Maybe also test_set here?
DATASETS = [
    "../data/train_pos_full.txt",
    "../data/train_neg_full.txt"
]

# These are two datasets I found when searching for bad word filters
BAD_WORD_LISTS = [
    "../data/Facebook Bad Words List - May 1, 2022.txt",
    "../data/youtube-blacklist-words_comma-separated-text-file.txt"
]



# Special cases found when browsing dataset
# $ -> s

def parse_bad_word_list(bad_word_list_file):
    """This parses badwords dataset, formatting words, and creating a list of bad words"""

    with open(bad_word_list_file, "rt") as f:
        # There's a few comments before actual content
        # Actual content is on single line so just stripping all lines with <1000 characters gives us the content we want
        bad_words = [line for line in f.readlines() if len(line) > 1000][0]

    # Strip space after comma
    bad_words = bad_words.replace(", ", ",")

    # Strip newline at end
    bad_words = bad_words.replace("\n", "")

    # Lowercase words
    bad_words = bad_words.lower()
    
    # Get list of badwords
    bad_words_list = bad_words.split(',')

    # Strip all words with digits containing more than alphabetical characters
    # We only want uncensored version
    bad_words_list = [x for x in bad_words_list if x.isalpha()]

    return bad_words_list


def create_set_of_bad_words():
    """Creates set of bad words"""
    full_bad_word_list = list()
    for bad_word_list_file in BAD_WORD_LISTS:
        full_bad_word_list+=parse_bad_word_list(bad_word_list_file)

    full_bad_word_list_set = set(full_bad_word_list)
    return full_bad_word_list_set

full_bad_word_list_set = create_set_of_bad_words()

# TODO manually strip some badwords here, like `dlck` or `fack`.

len(full_bad_word_list_set)
# %%

def contract_stars(sentence: str) -> str:
    return sentence.replace(" * ", "*")


def parse_twitter_dataset():

    # Read datasets
    all_tweets = list()
    for dataset in DATASETS:
        with open(dataset, "rt") as f:
            all_tweets+=f.readlines()

    # We assume our censor character is `*`, so only tweets with that character need to be looked at
    all_tweets_with_stars = [line for line in all_tweets if '*' in line]

    # Contract stars by removing surrounding spaces, i.e. `d * ck` -> `d*ck`
    all_tweets_with_stars_contracted = [contract_stars(line) for line in all_tweets_with_stars]
    all_tweets_with_stars_contracted

    all_words_set = set()
    for sentence in tqdm(all_tweets_with_stars_contracted):
        words = sentence.split(" ")
        [all_words_set.add(word) for word in words]

    all_words_set_only_stars = [word for word in all_words_set if '*' in word]
    all_words_set_only_stars

    # Only allow alphabetical characters and `*`
    # Strips stuff like `:*`
    all_words_set_only_stars_no_special = [word for word in all_words_set_only_stars if word.replace('*','').isalpha()]
    return all_words_set_only_stars_no_special

all_words_set_only_stars_no_special = parse_twitter_dataset()


# %%

# Cannot give pyspellchecker the string directly, it needs to read from file :c
with open("../data/badwords.txt", "wt") as f:
    f.writelines([word + '\n' for word in full_bad_word_list_set])

spell = SpellChecker(language=None, case_sensitive=True)
spell.word_frequency.load_text_file('../data/badwords.txt')

# %%

def get_number_of_stars_in_string(sentence):
    return sentence.count('*')

def build_uncensor_mapping():
    """Spellchecks censored words against bad word list to uncensor them"""
    spell_correct_mappings = set()
    for word in tqdm(all_words_set_only_stars_no_special):

        spell.distance = get_number_of_stars_in_string(word)
        corrected_word = spell.correction(word)

        if corrected_word != word and corrected_word in full_bad_word_list_set:
            spell_correct_mappings.add((word, corrected_word))

    return spell_correct_mappings

spell_correct_mappings = build_uncensor_mapping()

len(spell_correct_mappings)


# %%

def fake_preprocess_tokenize(word: str) -> str:
    """
    Adds spaces around `*` and adds leading and trailing space
    Basically turns `d*ck` into ` d * ck `
    (note the leading and trailing space)

    The reason for leading/trailing space is to later avoid matching cases like
    `SHITTY` -> `SH*TTY` -> `SH * TTY` -> `baSH * TTY` (capitalisation for emphasis)
    """
    return f" {word.replace('*', ' * ')} "


def get_full_dataset_as_list_of_strings():
    # Read datasets
    all_tweets = list()
    for dataset in DATASETS:
        with open(dataset, "rt") as f:
            all_tweets += f.readlines()
    return all_tweets
    

def filter_uncensor_mappings(spell_correct_mappings):
    """
    Post-process mappings by various critera to filter out potentially wrong mappings
    """
    # Filter out cases where spellchecker added/removed characters, those are likely bogus corrections
    reduced_spell_correct_mappings = [(word, suggested_correction) for (word, suggested_correction) in spell_correct_mappings if len(word) == len(suggested_correction)]

    # Filter out cases where the mappings barely appear in the dataset.
    # Those are either incorrect mappings that made it through on accident...
    # ...or if they are correct they appear so little it doesn't matter anyway.
    full_dataset_as_list_of_strings = get_full_dataset_as_list_of_strings()
    new_reduced_spell_correct_mappings = set()
    THRESHOLD_NUMBER_OF_APPEARENCES = 1
    for (word, suggested_correction) in tqdm(reduced_spell_correct_mappings):
    # for (word, suggested_correction) in reduced_spell_correct_mappings:
        original_version = fake_preprocess_tokenize(word)
        nr_of_occurences = len([sentence for sentence in full_dataset_as_list_of_strings if original_version in sentence])
        if nr_of_occurences >= THRESHOLD_NUMBER_OF_APPEARENCES:
            new_reduced_spell_correct_mappings.add((word, suggested_correction))
        # else:
        #     print(f"{original_version} |  {suggested_correction}  |  {nr_of_occurences}")

    return new_reduced_spell_correct_mappings

reduced_spell_correct_mappings = filter_uncensor_mappings(spell_correct_mappings)
# %%
print(json.dumps(dict(reduced_spell_correct_mappings), indent=4))


# %%
len(reduced_spell_correct_mappings)
# %%
# Mappings with added spaces to just use `some_string.replace()`
print(json.dumps(dict(
    [
        (fake_preprocess_tokenize(word), suggested_correction)
        for (word, suggested_correction) in reduced_spell_correct_mappings
    ]
), indent=4))

# %%
