# Debugging script to show the result of processing the tweets
from utils import read_args, normalized_log
from data_loader import TransformerDataLoader
import json

def remove_http_and_user(i: str) -> str:
    return i.replace("<url>","").replace("HTTPURL", "").replace("<user>", "").replace("@USER", "")

config = read_args()
negative_tweets, positive_tweets = TransformerDataLoader._load_tweets(
    config.preprocessing
)


negative_tweets.sort()
positive_tweets.sort()
normalized_log = list(filter(lambda x: remove_http_and_user(x[0]) != remove_http_and_user(x[1]), normalized_log))
normalized_log.sort()


with open(
    "processed-tweets.json",
    "w",
) as f:
    json.dump(
        {
            "negative_tweets": negative_tweets,
            "positive_tweets": positive_tweets,
            "config": config,
            "normalized_log": normalized_log,
        },
        f,
        indent=4,
    )

