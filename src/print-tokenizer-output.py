#!/usr/bin/env python3

import torch
import os
from utils import read_args
from tokenizer import PreTrainedTokenizer
from bunch import Bunch
from TweetNormalizer import normalizeTweet


def truncate_to_mask(id, mask):
    idx = 0
    while True:
        if mask[idx] == 0:
            break
        idx += 1

    res = id[:idx].tolist()
    # remove start and end markers
    assert res[0] == 0
    assert res[-1] == 2
    return res[1:-1]


inp = [
    "<3",
    # "^^",
    # ":)",
    # ":3",
    # ":d",
    # "=3",
    # ":/",
    # "</3",
    # ":(",
    # ":|",
    # "-.-",
    # ": )",
    # ": 3",
    # ": d",
    # "= 3",
    # ": (",
    # "< / 3",
    # ": |",
    # "- . -",
    # "shxt",
    # "shit",
    # "dick",
    # "d*ck",
    # "d * ck",
    # "test <3 <3 :)",
    # "foo  bar",
    # "foo bar",
    # "x",
    # "o",
    # "xx",
    "x" * 3,
    "x" * 4,
    # "x" * 5,
    # "x" * 6,
    # "x" * 7,
    # "xo",
    # "xoxo",
    # "xox",
    # "xoxox",
    # "xoxoxoxoxo",
    # "#cool",
    # "#fuckswithme",
    # "#sdjfasljkdfl",
    # "test afhjksdhjkds",
    "test 😂",
    "_ heart with arrow 💘 ",
    # "= [",
    # "=[",
    # "= ]",
    # "=]",
    # "-_-",
    # "- _ -",
    # "0-0",
    # "0 - 0",
    # "O - O",
    # "O-O",
    # "0_o",
    # "0 _ o",
    # ":')",
    # ":' )",
    # ": ' )",
    # ":'d",
    # ": 'd",
    # ": ' - (",
    # ":'-(",
    # ": ' 3",
    # ":'3",
    # "> _ >",
    # ">_>",
    # "=-0",
    # "=-O",
    # "=-o",
    # "= - o",
    # "=- o",
    # "=d",
    # "= d",
    "> . <",
    ">.<",
    # ": - ?",
    # ":-?",
    # ": $",
    # ":$",
    # "whooohoo woho whoho whoooho",
    # "\\ 123 \\ 474 hello",
    "don't take a the wrong way ... i'm just being friendly ...",
    "have a look - sony vaio vgn-fe 30b battery 53wh , 4800mah <url> xx",
    "damnnn that's impressiveeeee",
    "<user> heyyy thats at my birthday dinner !"
    " w/o",
    "backache",
    "something about him that got me like damnnn !",
    "I'll",
    " ur bootiful",
    "( r ) ( e ) ( t ) ( w ) ( e ) ( e ) ( t ) if you have less than 1o , ooo + followers ! ( f ) ( o ) ( l ) ( l ) ( o ) ( w ) - - - ) <user> and i'll help you gain ! ! ",
    "<user> fuck ! ! ! my phone wont let me use paypal grrr ! ! ! i'll just watch the video until i can get on a computer"
]

# inp += open(custom_dict_path).read().splitlines()
print(inp)


# This program shows the result of running `tokenize_tweets` on normalized and not normalized inputs
# It will print the output for each element of `inp` and if the result of the normalized tweet is different it also
# prints that next to it.
# Note that the results of tokenize_tweets always have a leading 0 and trailing 2 which are not printed here
def main() -> None:
    args = read_args()
    model_config = Bunch(args.model)
    preprocessing_config = Bunch(args.preprocessing)
    model_config.pretrained_model = "vinai/bertweet-base"
    print(model_config)
    print("")
    tokenizer = PreTrainedTokenizer(model_config)
    normalizedInp = list(
        map(lambda tweet: normalizeTweet(tweet, preprocessing_config), inp)
    )

    ids, masks = tokenizer.tokenize_tweets(inp)
    normalized_ids, normalized_masks = tokenizer.tokenize_tweets(normalizedInp)

    for i, a in enumerate(inp):
        before = truncate_to_mask(ids[i], masks[i])
        after = truncate_to_mask(normalized_ids[i], normalized_masks[i])
        if before != after:
            print('"' + a + '"', before, "\n  =>", '"' + normalizedInp[i] + '"', after)
        else:
            print('"' + a + '"', before)


if __name__ == "__main__":
    main()
