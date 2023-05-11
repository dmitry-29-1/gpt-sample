from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def build_tokenizer() -> Tokenizer:
    # Initialize a tokenizer
    result = Tokenizer(BPE(unk_token="[UNK]"))
    result.pre_tokenizer = Whitespace()

    # Initialize a trainer, training on the Alice in Wonderland text
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    result.train(files=["../../resources/datasets/alice_in_wonderland.txt"], trainer=trainer)

    # Save the tokenizer
    result.save("../../resources/datasets/alice_bpe_tokenizer.json")
    return result
