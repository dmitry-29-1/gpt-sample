from typing import List

from tokenizers import Tokenizer
from torch import tensor
from torch.utils.data import Dataset


class AliceDataset(Dataset):
    def __init__(self, data: List[int], sequence_len: int):
        self.data = data
        self.sequence_len = sequence_len

    def __len__(self):
        return len(self.data) - self.sequence_len

    def __getitem__(self, idx: int):
        return (
            tensor(self.data[idx: idx + self.sequence_len]),  # input sequence
            tensor(self.data[idx + 1: idx + self.sequence_len + 1]),  # target sequence
        )


def build_dataset(tokenizer: Tokenizer, sequence_len: int) -> AliceDataset:
    with open("../../resources/datasets/alice_in_wonderland.txt", "r") as f:
        lines = f.readlines()
        training_data: List[int] = [
            token
            for line in lines
            for token in tokenizer.encode(line).ids
        ]

    return AliceDataset(training_data, sequence_len=sequence_len)
