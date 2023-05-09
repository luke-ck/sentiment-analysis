from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
import tqdm
from bunch import Bunch
from transformers import PreTrainedTokenizerBase, AutoTokenizer


class Tokenizer(ABC):
    @abstractmethod
    def tokenize_tweets(self, tweets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class PreTrainedTokenizer(Tokenizer):
    def __init__(self, config: Bunch) -> None:
        # TODO: add our custom vocabulary to the model
        self.tokenizer = self.init_tokenizer(config.pretrained_model)
        self.max_token_length = config.max_token_length

    def init_tokenizer(self, pretrained_model: str) -> PreTrainedTokenizerBase:
        tokenizer_instance = AutoTokenizer.from_pretrained(pretrained_model)
        return tokenizer_instance

    def tokenize_tweets(self, tweets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # essentially this function does the same thing as the one below, just in batches, it's faster

        tokenized_input = self.tokenizer.batch_encode_plus(
            tweets,
            max_length=self.max_token_length,
            padding="max_length",
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return tokenized_input["input_ids"], tokenized_input["attention_mask"]

    def create_bert_input_features(self, docs: np.ndarray, max_seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: this function is slow as hell. Takes around 40s for 160k elems.
        # Maybe look at the implementation provided on the repo we are using as ref

        all_ids, all_masks = [], []
        for doc in tqdm.tqdm(docs, desc="Converting docs to features"):

            tokens = self.tokenizer.tokenize(doc)

            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:max_seq_length - 2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            masks = [1] * len(ids)

            # Zero-pad up to the sequence length.
            while len(ids) < max_seq_length:
                ids.append(0)
                masks.append(0)

            all_ids.append(ids)
            all_masks.append(masks)

        return torch.tensor(all_ids, dtype=torch.int64), torch.tensor(all_masks, dtype=torch.int64)


if __name__ == '__main__':
    args = Bunch({"tokenizer": "BertTokenizerFast", "max_token_length": 128, "pretrained_model": "bert-base-uncased"})
    tokenizer = PreTrainedTokenizer(args)
    ids, attention_mask = tokenizer.tokenize_tweets(["Hello, World!"])
    encoded_dict = tokenizer.tokenizer(["Hello, World!"], max_length=128,
                                       padding="max_length")
    ids_1, attention_mask_1 = encoded_dict["input_ids"], encoded_dict["attention_mask"]
    ids_2, attention_mask_2 = tokenizer.create_bert_input_features(np.array(["Hello, World!"]), 128)

    ids_1 = torch.tensor(ids_1)
    attention_mask_1 = torch.tensor(attention_mask_1)

    assert torch.all(torch.eq(ids, ids_1))
    assert torch.all(torch.eq(attention_mask, attention_mask_1))  # __call__ from tokenizer does the same thing!
    assert torch.all(torch.eq(ids, ids_2))
    assert torch.all(torch.eq(attention_mask, attention_mask_2))  # same shit here as well
