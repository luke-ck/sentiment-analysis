import json
from typing import Tuple, List
from time import localtime, strftime
import time
from emoji import demojize

# If set to True will dump information about how the leakage was cleaned up
DUMP_DEBUG = True


def count_unmatched_parens(tweet: str) -> Tuple[int, int]:
    unmatched_closing = 0
    unmatched_opening = 0

    for c in tweet:
        if c == ")":
            if unmatched_opening > 0:
                unmatched_opening -= 1
            else:
                unmatched_closing += 1

        if c == "(":
            unmatched_opening += 1
    return unmatched_opening, unmatched_closing


def last_unmatched_opening_pos(tweet: str) -> int:
    closing = 0

    for i, c in reversed(list(enumerate(tweet))):
        if c == ")":
            closing += 1
        if c == "(":
            if closing == 0:
                return i
            else:
                closing -= 1


pos_smileys = [":)", demojize("😊")]
neg_smileys = [":(", demojize("😕")]

# one_unmatched_opening_allowed_in determines the handling of single unmateched `(`
# set it to 1 to allow none
# set it to 1/2 to allow it in the last half of the tweet
# set it to 2/3 to allow it in the last third
# set it to 0.1 to allow it in the last 90%
# set it to -1 to allow it in the entire tweet
def dataleakage_class(tweet: str, one_unmatched_opening_allowed_in: float) -> str:

    has_pos_smiley = len([p for p in pos_smileys if p in tweet]) > 0
    has_neg_smiley = len([n for n in neg_smileys if n in tweet]) > 0

    if has_pos_smiley and has_neg_smiley:
        return "UNKNOWN"

    if has_pos_smiley and not has_neg_smiley:
        return "POS"

    if has_neg_smiley and not has_pos_smiley:
        return "NEG"

    opening, closing = count_unmatched_parens(tweet)

    if opening == 0 and closing == 0:
        return "OK"

    if opening > 0 and closing > 0:
        return "UNKNOWN"

    if opening > 0 and closing == 0:
        if opening == 1 and last_unmatched_opening_pos(tweet) > (
            len(tweet) * one_unmatched_opening_allowed_in
        ):
            return "OK"
        else:
            return "NEG"

    if closing > 0 and opening == 0:
        return "POS"

    raise Error("Unreachable")


def remove_parens(tweet: str) -> str:
    t = tweet.replace("(", "").replace(")", "")
    return " ".join(t.split())


def clean_data(leaky_neg, leaky_pos) -> Tuple[List[str], List[str]]:
    cleaned_neg = []
    cleaned_pos = []
    removed_unknown = []
    moved_from_pos_to_neg = []
    moved_from_neg_to_pos = []

    one_unmatched_opening_allowed_in = 0.5
    for neg_tweet in leaky_neg:
        clz = dataleakage_class(neg_tweet, one_unmatched_opening_allowed_in)
        if clz == "OK" or clz == "NEG":
            cleaned_neg.append(neg_tweet)
        elif clz == "POS":
            # moved_from_neg_to_pos.append(neg_tweet)
            # cleaned_pos.append(neg_tweet)
            removed_unknown.append(neg_tweet)
        elif clz == "UNKNOWN":
            removed_unknown.append(neg_tweet)
        else:
            raise Error("unreachable")

    for pos_tweet in leaky_pos:
        clz = dataleakage_class(pos_tweet, one_unmatched_opening_allowed_in)
        if clz == "OK" or clz == "POS":
            cleaned_pos.append(pos_tweet)
        elif clz == "NEG":
            # moved_from_pos_to_neg.append(pos_tweet)
            # cleaned_neg.append(pos_tweet)
            removed_unknown.append(pos_tweet)
        elif clz == "UNKNOWN":
            removed_unknown.append(pos_tweet)
        else:
            raise Error("unreachable")

    if DUMP_DEBUG:
        moved_from_pos_to_neg.sort()
        moved_from_neg_to_pos.sort()
        removed_unknown.sort()
        with open(
            "data-leakage-log"
            + strftime("%Y-%m-%d_%H:%M:%S", localtime())
            + "_"
            + str(time.time())
            + ".json",
            "w",
        ) as f:
            json.dump(
                {
                    "moved_from_pos_to_neg": moved_from_pos_to_neg,
                    "moved_from_neg_to_pos": moved_from_neg_to_pos,
                    "removed_unknown": removed_unknown,
                },
                f,
                indent=4,
            )
    cleaned_neg = list(map(remove_parens, cleaned_neg))
    cleaned_pos = list(map(remove_parens, cleaned_pos))

    print(
        f"While removing the data leakage {len(moved_from_pos_to_neg) + len(moved_from_neg_to_pos)} tweets where moved and {len(removed_unknown)} tweets were removed from the dataset"
    )
    return cleaned_neg, cleaned_pos
