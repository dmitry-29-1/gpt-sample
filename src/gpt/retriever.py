import torch
from tokenizers import Tokenizer
from torch import tensor
from torch.distributions.categorical import Categorical

from gpt.model import GPT


class Retriever:
    def __init__(self, tokenizer: Tokenizer, model: GPT):
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, text: str, temperature: float = 1.0) -> str:
        with torch.no_grad():
            tokens = list(self.tokenizer.encode(text).ids)
            if len(tokens) > self.model.sequence_len:
                raise ValueError(
                    f"Input text length ({len(tokens)}) exceeds the model limit ({self.model.sequence_len}")

            as_string = self.tokenizer.decode(tokens)
            for _ in range(50):
                inputs = tensor(tokens).unsqueeze(0)
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs / temperature, dim=-1)
                next_token = Categorical(probabilities).sample()[:, -1].item()
                tokens.append(next_token)
                as_string = self.tokenizer.decode(tokens)
                if next_token == self.tokenizer.token_to_id("[EOS]"):
                    break
            return as_string
