import torch
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define tokenizer and special tokens
tokenizer = get_tokenizer("basic_english")
special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]

# Load Wikitext-2 dataset
train_data, valid_data, test_data = WikiText2.splits(tokenizer=tokenizer, special_tokens=special_tokens)

# Build vocabulary
vocabulary = build_vocab_from_iterator(iter(train_data), specials=special_tokens)
vocabulary.set_default_index(vocabulary["<unk>"])

# Define dataset
class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, data, vocabulary):
        self.data = data
        self.vocabulary = vocabulary

    def __getitem__(self, index):
        return torch.tensor([self.vocabulary[token] for token in self.data[index]], dtype=torch.long)

    def __len__(self):
        return len(self.data)

# Define batch function
def collate_batch(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

# Create DataLoader
batch_size = 32
train_dataset = WikiTextDataset(train_data, vocabulary)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
