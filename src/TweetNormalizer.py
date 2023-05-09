# This file is based on from https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
# under MIT License
#
# This file includes changes from the original above to addapt it for
# our specific dataset

import math
import nltk
import re
import pkg_resources

from nltk.tokenize import TweetTokenizer
from utils import project_root
from os import path
from functools import lru_cache
from gibberish_detector import detector
from emoji import demojize
from symspellpy import SymSpell, Verbosity
from bunch import Bunch
from utils import normalized_log

from TweetNormalizerData import (
    abbreviations,
    global_regexes,
    per_token_regexes,
    global_replacement_map,
    curse_word_uncensor,
)

tokenizer = TweetTokenizer()
stemmer = nltk.stem.SnowballStemmer("english")


# All in one class for spellchecking, Autocorrect and gibberish detection
class SpellingHelper:
    def __init__(self, config: Bunch) -> None:
        self.config = config
        self.check_spelling = self.config['check_spelling']

        self._sym_spell = SymSpell(
            max_dictionary_edit_distance=self.config.symspell_max_dictionary_edit_distance,
            prefix_length=max(7, self.config.symspell_max_dictionary_edit_distance + 1),
        )
        sym_dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        # term_index is the column of the term and count_index is the
        # column of the term frequency
        self._sym_spell.load_dictionary(
            sym_dictionary_path, term_index=0, count_index=1
        )

        bertweet_dict_path = path.join(
            project_root, config.data_path, "bertweet-dict-lower-sorted.txt"
        )
        with open(bertweet_dict_path, "r") as dict_file:
            self._bert_dict = set(dict_file.read().splitlines())

        # "gibberish-detector-lower.model" was created by
        # gibberish-detector train big-lower.txt > gibberish-detector-lower.model
        # where big-lower.txt is
        # tr '[:upper:]' '[:lower:]' < big.txt > big-lower.txt
        # TODO threshhold from config
        self._gibberish_detector = detector.create_from_model(
            path.join(project_root, config.data_path, "gibberish-detector-lower.model"),
            limit=config.gibberish_detector_limit
        )

    @lru_cache(maxsize=None)
    def correct_spelling(self, word: str) -> str:
        ed = max(1, math.ceil(len(word) / self.config.symspell_letters_per_typo))
        ed = min(ed, self.config.symspell_max_dictionary_edit_distance)

        suggestions = self._sym_spell.lookup(
            word, Verbosity.CLOSEST, max_edit_distance=ed, include_unknown=True
        )

        return suggestions[0].term

    # Determines if a given word is gibberish
    def is_gibberish(self, word: str) -> bool:
        return self._gibberish_detector.is_gibberish(word)

    # Check if the given word  in the dictionary of known bertweet words
    def is_bertweet_word(self, word: str):
        return word in self._bert_dict


# For an input string return all possible combinations of adding single spaces to it
#
# spaceout("foo") == ['foo', 'fo o', 'f oo', 'f o o']
def spaceout(inp: str) -> [str]:
    if len(inp) <= 1:
        return [inp]
    if len(inp) == 2:
        return [inp, inp[0] + " " + inp[1]]
    sub = spaceout(inp[1:])
    a1 = map(lambda p: inp[0] + p, sub)
    a2 = map(lambda p: inp[0] + " " + p, sub)
    return list(a1) + list(a2)


# remove_repeated_letters("helllooooo") == "helo"
# remove_repeated_letters("awwwwessssommmee") == "awesome"
def remove_repeated_letters(token):
    res = token[0]
    for l in token:
        if l != res[-1]:
            res += l
    return res


# This is the core "algorithm" of the normalization.
# If a token is alpha and not in the bertweet dictionary we use this
# to try to get some usable token out of it by using a combination of autocorrect,
# stemming and removing repated letters
def try_to_make_alpha_token_usable(token, spelling_helper):
    first_stem = stemmer.stem(token)
    if spelling_helper.is_bertweet_word(first_stem):
        return first_stem

    if spelling_helper.check_spelling:
        token = spelling_helper.correct_spelling(token)

    first_correct = token
    if spelling_helper.is_bertweet_word(token):
        return token

    token = stemmer.stem(token)
    if spelling_helper.is_bertweet_word(token):
        return token

    if spelling_helper.check_spelling:
        token = spelling_helper.correct_spelling(token)
        if spelling_helper.is_bertweet_word(token):
            return token

    token = remove_repeated_letters(token)
    if spelling_helper.is_bertweet_word(token):
        return token

    if spelling_helper.check_spelling:
        token = spelling_helper.correct_spelling(token)
        if spelling_helper.is_bertweet_word(token):
            return token

    if spelling_helper.is_gibberish(first_correct):
        return ""

    return first_correct


def normalizeToken(token, spelling_helper):
    res = _normalizeToken(token, spelling_helper)
    return res


dotted_word_re = re.compile("^([a-z]{2,}([\\.-]))+[a-z]{3,}$")


@lru_cache(maxsize=None)
def _normalizeToken(token, spelling_helper):
    token = re.sub(r'(\w{2,})(\w)(\2{2,})$', r'\1\2', token)
    if token == "<user>":
        return "@USER"

    if token == "<url>" or token.startswith("http") or token.startswith("www"):
        return "HTTPURL"

    if token in abbreviations:
        return abbreviations[token]

    for expr, replacement in per_token_regexes.items():
        if bool(expr.match(token)):
            return replacement

    if len(token) == 1:
        return demojize(token)

    bertweet_knows_this = spelling_helper.is_bertweet_word(token)

    dotted_match = dotted_word_re.match(token)
    if bool(dotted_match) and not bertweet_knows_this:
        subs = token.split(dotted_match.group(2))
        return (" " + dotted_match.group(2) + " ").join(
            map(lambda tok: normalizeToken(tok, spelling_helper), subs)
        )

    if token.isalpha() and not bertweet_knows_this:
        return try_to_make_alpha_token_usable(token, spelling_helper)

    if bertweet_knows_this:
        return token

    if spelling_helper.is_gibberish(token):
        return ""

    return token


class SpellingHelperBuilder:
    def __init__(self):
        self._cached_helper = None
        self._cached_config = None

    def __call__(self, config):
        new_frozen_config = tuple(sorted(config.items()))
        if self._cached_config != new_frozen_config:
            self._cached_config = new_frozen_config
            print("Creating a new SpellingHelper for a new config with hash", hash(new_frozen_config))
            self._cached_helper = SpellingHelper(config)
        return self._cached_helper


spelling_helper_cache = SpellingHelperBuilder()


def normalizeTweet(tweet: str, config: Bunch) -> str:
    res = _normalizeTweet(tweet, config)
    normalized_log.append([tweet, res])
    return res


def _normalizeTweet(tweet, config: Bunch):
    """
    adapted from https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
    """
    uncensor_swearwords = config['uncensor_swearwords']
    translate_emoticons_to_emoji = config['translate_emoticons_to_emoji']
    normalize = config['normalize']

    if uncensor_swearwords:
        # Replace swearwords here according to static mapping
        for pattern, replacement in curse_word_uncensor.items():
            tweet = tweet.replace(pattern, replacement)

    for expr, replacement in global_regexes.items():
        tweet = re.sub(expr, replacement, tweet)

    tokens = tokenizer.tokenize(tweet)
    if normalize:
        spelling_helper = spelling_helper_cache(config)
        normTweet = " ".join([normalizeToken(token, spelling_helper) for token in tokens])

        if normTweet.endswith("( @USER live on HTTPURL"):
            normTweet += " )"

        normTweet = (
            normTweet.replace("cannot ", "can not ")
                .replace("n't ", " n't ")
                .replace("n 't ", " n't ")
                .replace("ca n't", "can't")
                .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
                .replace("'re ", " 're ")
                .replace("'s ", " 's ")
                .replace("'ll ", " 'll ")
                .replace("'d ", " 'd ")
                .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
                .replace(" p . m ", " p.m ")
                .replace(" a . m .", " a.m.")
                .replace(" a . m ", " a.m ")
        )
        normTweet = (
            normTweet.replace("? ? ?", "?")
                .replace("! ! !", "!")
                .replace("! !", "!")
                .replace("> > >", "> >")
                .replace("? ?", "?")
                .replace("- - -", "-")
                .replace("/ / ", "/")
                .replace(" ` ", "'")
                .replace("---", "-")
        )

        tweet = " ".join(normTweet.split())
    else:
        tweet = " ".join(tweet.split())
    return tweet


if __name__ == "__main__":
    from utils import read_args

    args = read_args()
    config = Bunch(args.preprocessing)

    print(
        normalizeTweet(
            "SC has first <user> two <url> presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier",
            config,
        )
    )
    """
    assert is_xoxo("xo") == True
    assert is_xoxo("xox") == True
    assert is_xoxo("hello") == False
    assert is_xoxo("xoxo") == True
    assert is_xoxo("xoxoxoxoxox") == True
    assert is_xoxo("xoxoxoxoxoxn") == False
    print(normalizeTweet("This is great xoxoxoxoxoxoxoxo 😂"))
    assert (
        normalizeTweet("This is great xoxoxoxoxoxoxoxo 😂")
        == "This is great hugs and kisses :face_with_tears_of_joy:"
    )
    """
    from data_loader import BaseDataLoader
    import json

    try:

        dat = BaseDataLoader._load_tweets_file("./data/train_pos_full.txt", config)
        print(len(dat))
        with open("normalized-pos.txt", "w") as f:
            f.write("\n".join(dat))
        dat = BaseDataLoader._load_tweets_file("./data/train_neg_full.txt", config)
        print(len(dat))
        with open("normalized-neg.txt", "w") as f:
            f.write("\n".join(dat))
    except KeyboardInterrupt:
        pass

    json.dump(
        {k: v for k, v in corrections.items() if v[1] > 0},
        open("corrections.json", "w"),
        indent=4,
        sort_keys=True,
    )
    print("wrote json")
