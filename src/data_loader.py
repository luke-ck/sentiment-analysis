from tokenizer import Tokenizer, PreTrainedTokenizer
from bunch import Bunch
from abc import ABC
from utils import project_root
from os import path
from typing import List, Tuple
import torch
import numpy as np
import os
from tqdm.contrib import tmap
from torch.utils.data import TensorDataset
from TweetNormalizer import normalizeTweet
import pathlib
from data_leakage import clean_data, remove_parens


class BaseDataLoader(ABC):
    """ Abstract Interface for data loading. Contains static methods
    that deal with loading tweets and generating labels. All the methods
    that don't need an instance go here.
    """

    @staticmethod
    def _load_tweets_file(filepath: str, config: dict, normalize=True) -> List[str]:
        """
        base function for loading the tweet text files. 
        Returns a list of unique tweets

        Args:
            filepath (str): path to tweets dataset. typically under data/

        Returns:
            List[str]: list of tweets
        """
        config = Bunch(config)
        with open(filepath) as f:
            contents = f.read()
            lines = contents.splitlines()
            unique_lines = set(lines)
            if normalize:
                unique_lines = tmap(lambda tweet: normalizeTweet(tweet, config), unique_lines,
                                    desc="normalizing tweets")

            unique_lines = list(unique_lines)
            print(f"Loaded  {len(lines)} tweets from {filepath}. " +
                  f"The data contained {len(lines) - len(unique_lines)} duplicates. " +
                  f"Thus there is a total of {len(unique_lines)} unique tweets.")
            return unique_lines

    @staticmethod
    def _load_tweets(config: dict) -> Tuple[List, List]:
        data_path = path.join(project_root, config["data_path"])

        if config["use_full_train_data"]:
            neg_file = config["train_neg_full_file_name"]
            pos_file = config["train_pos_full_file_name"]
        else:
            neg_file = config["train_neg_small_file_name"]
            pos_file = config["train_pos_small_file_name"]

        full_neg_path = path.join(data_path, neg_file)
        full_pos_path = path.join(data_path, pos_file)

        clean_before = config["clean_data_leakage_before_normalization"]
        clean_after = config["clean_data_leakage_after_normalization"]
        normalize = config["normalize"]
        clean = config['clean_data_leakage']
        leaky_neg = BaseDataLoader._load_tweets_file(full_neg_path, config, normalize=normalize)
        leaky_pos = BaseDataLoader._load_tweets_file(full_pos_path, config, normalize=normalize)

        if clean:
            if clean_before:
                config_bunch = Bunch(config)
                leaky_neg, leaky_pos = clean_data(leaky_neg, leaky_pos)
                leaky_neg = list(tmap(lambda tweet: normalizeTweet(tweet, config_bunch), leaky_neg,
                                      desc="normalizing neg tweets after cleaning data leaks"))
                leaky_pos = list(tmap(lambda tweet: normalizeTweet(tweet, config_bunch), leaky_pos,
                                      desc="normalizing pos tweets after cleaning data leaks"))

            if clean_after:
                return clean_data(leaky_neg, leaky_pos)

        return leaky_neg, leaky_pos

    @staticmethod
    def load_test_tweets(config: dict) -> np.ndarray:

        data_path = path.join(project_root, config["data_path"])
        test_file = config["test_data"]
        full_test_path = path.join(data_path, test_file)
        with_parens = BaseDataLoader._load_tweets_file(full_test_path, config)
        without_parens = list(map(remove_parens, with_parens))

        return np.array(without_parens)

    @staticmethod
    def load_tweets(config: dict) -> Tuple[List, List]:
        """ 
        This function loads unique tweets under data/ and returns a tuple of
        positive tweets and negative tweets 

        Args:
            config (dict): this should be dictionary found under "model" in the json config

        Returns:
            Tuple[List, List]: unique positive and negative tweets. Not processed
        """
        pos_tweets, neg_tweets = BaseDataLoader._load_tweets(config)
        # TODO: log total number of tweets, maybe other stats
        return pos_tweets, neg_tweets

    @staticmethod
    def generate_labels(total_neg: int, total_pos: int) -> np.ndarray:
        return np.concatenate([
            np.zeros(total_neg, dtype=np.int64),
            np.ones(total_pos, dtype=np.int64),
        ]
        )


class TransformerDataLoader(BaseDataLoader):
    """
    This class deals with everything related to data loading 
    and generating the dataset that is used by the base model, and
    implements convenience features such as batching through Torch's 
    DataLoader.
    This class gets passed a tokenizer that is instantiated, hence 
    the intended usage is for it to be instantiated, for example 
    when creating the train-validation split
    TODO: do we include the preprocessor here too? Or we keep that as
    as a separate stage
    TODO: switch to Bunch from dict
    """

    def __init__(self, config: Bunch, tokenizer: Tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def train_validation_split(
            self, validation_size: float, random_seed: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert 0 <= validation_size <= 1

        tweets, labels = self._get_tweets_and_labels()
        total_validation = int(validation_size * len(tweets))
        total_train = len(tweets) - total_validation

        shuffled_indices = np.random.permutation(len(tweets))
        train_indices = shuffled_indices[:total_train]
        val_indices = shuffled_indices[total_train:]

        print(f"Using {len(train_indices)} train and {len(val_indices)} validation samples")

        return tweets[train_indices], tweets[val_indices], labels[train_indices], labels[val_indices]

    def _get_tweets_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        negative_tweets, positive_tweets = TransformerDataLoader._load_tweets(self.config.preprocessing)
        # len_negative_tweets = int(len(negative_tweets) / 3)
        # len_positive_tweets = int(len(positive_tweets) / 3)
        all_tweets = negative_tweets[:len(negative_tweets)] + positive_tweets[:len(positive_tweets)]

        labels = TransformerDataLoader.generate_labels(len(negative_tweets), len(positive_tweets))
        return np.array(all_tweets), labels

    def create_datasets(self) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        train_data_cache_path = self.config.preprocessing["train_data_cache_path"]
        val_data_cache_path = self.config.preprocessing["val_data_cache_path"]
        test_data_cache_path = self.config.preprocessing["test_data_cache_path"]

        cache_paths = [train_data_cache_path, val_data_cache_path, test_data_cache_path]
        if "DONT_LOAD_FROM_CACHE" in os.environ:
            print(
                "DONT_LOAD_FROM_CACHE environment variable is set: Not trying to use the cache for create_datasets and will store the newly computed datasets in the cache")
        elif all(map(path.exists, cache_paths)):
            print("Dataset already exists in cache. Loading from there")
            return map(torch.load, cache_paths)

        print(f"Starting creation of the datasets for validation and training...")
        seed = self.config.preprocessing["seed"]
        max_seqlen = self.config.model["max_token_length"]
        validation_size = self.config.preprocessing["validation_size"]
        train_tweets, val_tweets, train_labels, val_labels = self.train_validation_split(validation_size, seed)
        test_tweets = TransformerDataLoader.load_test_tweets(self.config.preprocessing)

        train_token_ids, train_attn_mask = self.tokenizer.create_bert_input_features(train_tweets, max_seqlen)
        val_token_ids, val_attn_mask = self.tokenizer.create_bert_input_features(val_tweets, max_seqlen)
        test_token_ids, test_attn_mask = self.tokenizer.create_bert_input_features(test_tweets, max_seqlen)

        train_data = TensorDataset(train_token_ids, train_attn_mask, torch.from_numpy(train_labels))
        val_data = TensorDataset(val_token_ids, val_attn_mask, torch.from_numpy(val_labels))
        test_data = TensorDataset(test_token_ids, test_attn_mask)
        # TODO: this is kinda bullshit. It is not entirely obvious what type the arrays in this function are
        # I would expect them to have the same type i.e. numpy array or torch tensors

        # TODO: do the rest of the preprocessing like removing data leaks here.
        print("Finished creating the datasets...")

        # Create cache folder if it doesn't exist already
        for cache_path in [train_data_cache_path, val_data_cache_path, test_data_cache_path]:
            pathlib.Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)

        torch.save(train_data, train_data_cache_path)
        torch.save(val_data, val_data_cache_path)
        torch.save(test_data, test_data_cache_path)

        return train_data, val_data, test_data


if __name__ == "__main__":
    preprocessing = dict({
        "data_path": "./data",
        "train_neg_small_file_name": "train_neg.txt",
        "train_pos_small_file_name": "train_pos.txt",
        "train_neg_full_file_name": "train_neg_full.txt",
        "train_pos_full_file_name": "train_pos_full.txt",
        "test_data": "test_data.txt",
        "use_full_train_data": False,
        "seed": 42,
        "validation_size": 0.1
    })
    model = Bunch(dict({
        "tokenizer": "DistilBertTokenizer",
        "max_token_length": 140,
        "pretrained_model": "distilbert-base-uncased",
        "batch_size": 32
    }))
    tokenizer = PreTrainedTokenizer(model)
    data_loader = TransformerDataLoader(preprocessing, tokenizer)
    data_loader.create_datasets()
