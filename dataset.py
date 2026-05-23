"""
dataset.py — Multi30k dataset utilities for DA6401 Assignment 3.

This file intentionally avoids torchtext. It uses only datasets, spacy and torch.
German is treated as source language and English as target language.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

try:
    import spacy
except Exception:  # spacy is required only when the dataset/tokenizer is constructed
    spacy = None


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def _load_spacy_tokenizer(lang: str):
    """Load a small spaCy tokenizer if available; otherwise use a blank tokenizer."""
    model_names = {
        "de": ["de_core_news_sm", "de"],
        "en": ["en_core_web_sm", "en"],
    }
    if spacy is None:
        raise ImportError("spacy is required for tokenization. Install it with: pip install spacy")
    for name in model_names.get(lang, [lang]):
        try:
            return spacy.load(name, disable=["tagger", "parser", "ner", "lemmatizer"])
        except Exception:
            pass
    return spacy.blank(lang)


@dataclass
class SimpleVocab:
    stoi: dict[str, int]
    itos: list[str]
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[self.unk_token])

    def lookup_token(self, index: int) -> str:
        if 0 <= int(index) < len(self.itos):
            return self.itos[int(index)]
        return self.unk_token

    def lookup_indices(self, tokens: Sequence[str]) -> list[int]:
        unk = self.stoi[self.unk_token]
        return [self.stoi.get(tok, unk) for tok in tokens]

    def to_serializable(self) -> dict:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
        }

    @classmethod
    def from_serializable(cls, data: dict) -> "SimpleVocab":
        return cls(
            stoi=dict(data["stoi"]),
            itos=list(data["itos"]),
            unk_token=data.get("unk_token", "<unk>"),
            pad_token=data.get("pad_token", "<pad>"),
            sos_token=data.get("sos_token", "<sos>"),
            eos_token=data.get("eos_token", "<eos>"),
        )


def build_simple_vocab(
    token_stream: Iterable[Iterable[str]],
    min_freq: int = 2,
    max_size: Optional[int] = None,
) -> SimpleVocab:
    counter = Counter()
    for tokens in token_stream:
        counter.update(tokens)

    itos = list(SPECIAL_TOKENS)
    max_extra = None if max_size is None else max(0, max_size - len(itos))
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in SPECIAL_TOKENS:
            continue
        if max_extra is not None and len(itos) - len(SPECIAL_TOKENS) >= max_extra:
            break
        itos.append(token)

    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return SimpleVocab(stoi=stoi, itos=itos)


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[SimpleVocab] = None,
        tgt_vocab: Optional[SimpleVocab] = None,
        min_freq: int = 2,
        max_vocab_size: int = 12_000,
        max_len: int = 100,
        lower: bool = True,
    ):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        split_map = {
            "valid": "validation",
            "val": "validation",
            "dev": "validation",
        }
        self.split = split_map.get(split, split)
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.lower = lower

        self.de_tokenizer = _load_spacy_tokenizer("de")
        self.en_tokenizer = _load_spacy_tokenizer("en")

        try:
            from datasets import load_dataset
        except Exception as exc:
            raise ImportError("datasets is required to load Multi30k. Install it with: pip install datasets") from exc

        self.raw_dataset = load_dataset("bentrevett/multi30k", split=self.split)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []

        if self.src_vocab is not None and self.tgt_vocab is not None:
            self.process_data()

    def _extract_pair(self, example) -> tuple[str, str]:
        """Return (German, English), robust to common HF field formats."""
        if "de" in example and "en" in example:
            return example["de"], example["en"]
        if "translation" in example:
            tr = example["translation"]
            return tr.get("de", tr.get("deu", "")), tr.get("en", tr.get("eng", ""))
        if "german" in example and "english" in example:
            return example["german"], example["english"]
        raise KeyError(f"Could not find German/English fields in example keys: {list(example.keys())}")

    def _tok_de(self, text: str) -> list[str]:
        if self.lower:
            text = text.lower()
        return [tok.text.strip() for tok in self.de_tokenizer.tokenizer(text) if tok.text.strip()]

    def _tok_en(self, text: str) -> list[str]:
        if self.lower:
            text = text.lower()
        return [tok.text.strip() for tok in self.en_tokenizer.tokenizer(text) if tok.text.strip()]

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_tokens = []
        tgt_tokens = []
        for ex in self.raw_dataset:
            de, en = self._extract_pair(ex)
            src_tokens.append(self._tok_de(de))
            tgt_tokens.append(self._tok_en(en))

        self.src_vocab = build_simple_vocab(src_tokens, min_freq=self.min_freq, max_size=self.max_vocab_size)
        self.tgt_vocab = build_simple_vocab(tgt_tokens, min_freq=self.min_freq, max_size=self.max_vocab_size)
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError("Call build_vocab() first, or pass src_vocab and tgt_vocab to the constructor.")

        examples = []
        for ex in self.raw_dataset:
            de, en = self._extract_pair(ex)
            src_tokens = self._tok_de(de)
            tgt_tokens = self._tok_en(en)

            # Reserve room for <sos> and <eos>.
            src_tokens = src_tokens[: max(0, self.max_len - 2)]
            tgt_tokens = tgt_tokens[: max(0, self.max_len - 2)]

            src_ids = [SOS_IDX] + self.src_vocab.lookup_indices(src_tokens) + [EOS_IDX]
            tgt_ids = [SOS_IDX] + self.tgt_vocab.lookup_indices(tgt_tokens) + [EOS_IDX]

            examples.append((torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)))

        self.examples = examples
        return self.examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]

    def collate_fn(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
        return src_padded, tgt_padded
