from collections import Counter

import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

UNK, PAD, SOS, EOS = 0, 1, 2, 3
SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]


class Vocab:
    def __init__(self, tokens, min_freq=2):
        counts = Counter(tokens)
        words = [w for w, c in counts.items() if c >= min_freq and w not in SPECIALS]
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, word):
        return self.stoi.get(word, UNK)

    def lookup_token(self, idx):
        return self.itos[idx]


class Multi30kDataset(Dataset):
    def __init__(self, split="train", src_vocab=None, tgt_vocab=None, min_freq=2):
        split = "validation" if split in {"val", "valid"} else split
        self.de_tok = spacy.blank("de").tokenizer
        self.en_tok = spacy.blank("en").tokenizer

        data = load_dataset("bentrevett/multi30k", split=split)
        self.src_texts, self.tgt_texts = self._read_columns(data)

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab(min_freq)
        else:
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

        self.src_data, self.tgt_data = self.process_data()

    @staticmethod
    def _read_columns(data):
        if "de" in data.column_names and "en" in data.column_names:
            return list(data["de"]), list(data["en"])
        trans = data["translation"]
        return [x["de"] for x in trans], [x["en"] for x in trans]

    def tokenize_de(self, text):
        return [t.text.lower() for t in self.de_tok(text)]

    def tokenize_en(self, text):
        return [t.text.lower() for t in self.en_tok(text)]

    def build_vocab(self, min_freq=2):
        src = [t for s in self.src_texts for t in self.tokenize_de(s)]
        tgt = [t for s in self.tgt_texts for t in self.tokenize_en(s)]
        return Vocab(src, min_freq), Vocab(tgt, min_freq)

    def encode(self, tokens, vocab):
        ids = [SOS] + [vocab[t] for t in tokens] + [EOS]
        return torch.tensor(ids, dtype=torch.long)

    def process_data(self):
        src = [self.encode(self.tokenize_de(s), self.src_vocab) for s in self.src_texts]
        tgt = [self.encode(self.tokenize_en(s), self.tgt_vocab) for s in self.tgt_texts]
        return src, tgt

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def collate_fn(batch):
    src, tgt = zip(*batch)
    return (
        pad_sequence(src, batch_first=True, padding_value=PAD),
        pad_sequence(tgt, batch_first=True, padding_value=PAD),
    )
