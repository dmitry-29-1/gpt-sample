# Summary

This README file provides instructions on how to design and train a simple GPT LLM. This provides an example of a model that predicts the next word after trained on "Alice's Adventures in Wonderland" text.  

## Quick guide

Run `src/gpt/train.py`, tweak as needed. 

## Project structure

I may convert this into a Jupyter notebook soon, but for now in form of a project.

- `src/gpt/dataset.py` — data set wrapper;
- `src/gpt/logs.py` — utility for logging;
- `src/gpt/model.py` — implements architecture from GPT paper [https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf);
- `src/gpt/retriever.py` — retrieving interface for the LLM;
- `src/gpt/tokenizer.py` — tokenizer logic;
- `src/gpt/train.py` — training script;

## Training walk-through

This covers steps from `src/gpt/train.py` that train the model in details.

1. Import the necessary dependencies:
```python
import os
import time
import torch
from tokenizers import Tokenizer
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from gpt.dataset import build_dataset
from gpt.logs import log_batch
from gpt.model import GPT
from gpt.retriever import Retriever
from gpt.tokenizer import build_tokenizer
```

2. Set the desired parameters:
```python
total_epochs = 3
batch_size = 16

embed_dim = 768
sequence_len = 128
num_layers = 24
num_heads = 12
forward_expansion = 4

weights_filepath = f"../../resources/gpt/weights_ed{embed_dim}_s{sequence_len}_l{num_layers}_h{num_heads}_f{forward_expansion}.pth"
```

3. Initialize the tokenizer:
```python
tokenizer: Tokenizer = build_tokenizer()
```

4. Initialize the model:
```python
model = GPT(
    vocab_size=tokenizer.get_vocab_size(),
    sequence_len=sequence_len,
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    forward_expansion=4,
    dropout=0.1,
    pad_idx=tokenizer.token_to_id("[PAD]")
)
if os.path.exists(weights_filepath):
    model.load_state_dict(torch.load(weights_filepath))
```

5. Build the dataset and dataloader:
```python
dataset: Dataset = build_dataset(tokenizer, sequence_len=sequence_len)
dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

6. Define the loss function and optimizer:
```python
optimizer = Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
```

7. Move the model to GPU if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

8. Test the model before training:
```python
retriever = Retriever(tokenizer, model)
answer = retriever.generate("what is your name?")
```

9. Train the model:
```python
start = time.time()
for epoch_index in range(total_epochs):
    for batch_index, batch in enumerate(dataloader):
        inputs, targets = batch

        # Move data to GPU if available
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_batch(loss.item(), start, epoch_index, total_epochs, batch_index, len(dataloader))

    torch.save(model.state_dict(), weights_filepath)
    print("Saved")
```



