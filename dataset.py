"""
dataset.py — Multi30k dataset utilities

Only datasets, spacy and torch are used.
German is the source languange, English is the target langyage
"""

# lets use type hints more flexibly
# can refer to a class inside its own definition without errors
# example: class Node:
#           next: Node
from __future__ import annotations

# used to count how many times each item appears
from collections import Counter
# example: words = ["hello", "hi", "hello"]
# Counter(words) outputs {"hello": 2, "hi": 3}

# Used to quickly create simple classes that store data
from dataclasses import dataclass
# using @dataclass wrapper internally does init, repr etc. without having to
# separately write the constructor each time
# recal w rapper is a piece of coe that wraps around another function or object
# to modify its behaviour without changing its original source code

# used to make the code more readable 
from typing import Iterable, List, Optional, Sequence, Tuple

# pytorch is used for tensors, making models, GPU usage, saving and loading modelss
import torch
# used to pad variable length sequences so they become the same length
from torch.nn.utils.rnn import pad_sequence

# its an abstract base class used to store and organize the training data
# used to create own custom dataset class - example: class TranslationDataset(Dataset)
# usually overwrite __len__ and __getitem__
# pytorch can then use it with DataLoader for batching, shuffling, training,....
from torch.utils.data import Dataset


try:
    import spacy
except Exception:  # spacy is needed when the tokenizer is made
    spacy = None


# define the special tokens used by the transfromer
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX = 0 # unknown word
PAD_IDX = 1 # used to make all sentences in a batch same length
SOS_IDX = 2 # start of sentence
EOS_IDX = 3 # end of sentence


def _load_spacy_tokenizer(lang: str):
    # load german and english tokenizers else load blank tokenizer
    model_names = {
        "de": ["de_core_news_sm", "de"],
        "en": ["en_core_web_sm", "en"],
    }
    if spacy is None:
        raise ImportError("spacy is required")
    for name in model_names.get(lang, [lang]):
        try:
            return spacy.load(name, disable=["tagger", "parser", "ner", "lemmatizer"])
        except Exception:
            pass
    # if nothing was loadable then initialize it to blank
    return spacy.blank(lang)


@dataclass # using the wrapper
# so skipping over writing __init__...
# use this to get numbers for words
class SimpleVocab:
    # stoi is string to index - get corresponding integer from the word/token
    stoi: dict[str, int]
    itos: list[str] # itos is the reverse
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"

    # return the vocab size - needed when making the transformer
    def __len__(self) -> int:
        return len(self.itos)
     
    # check if a word exists in the vocab
    # "man" in vocab does vocab.__contains__("man")
    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    # if the word exists return its index
    # can just do vocab["man"] instead of vocab.stoi["man"]
    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[self.unk_token])

    # does the reverse conversion of integer_ID to word
    # vocab.lookup_token(5)
    # useful in inference - model outputs token IDs and we convert them back to english words
    def lookup_token(self, index: int) -> str:
        if 0 <= int(index) < len(self.itos):
            return self.itos[int(index)]
        return self.unk_token

    # converts a full list of wors into IDs
    # tokens = ["a", "man", "runs"]
    # vocab.lookup_indices(tokens)
    def lookup_indices(self, tokens: Sequence[str]) -> list[int]:
        unk = self.stoi[self.unk_token]
        return [self.stoi.get(tok, unk) for tok in tokens]

    # converts the vocab object into a normal python dictionary
    # cuz we need to save the vocab inside the checkpoint
    # pyTorch checkpoints save dictionaries easily
    def to_serializable(self) -> dict:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
        }

    # does the opposite
    # takes a saved dictionary and rebuilds the SimpleVocab object
    # class method means the function belongs to the class itself
    # not to an already created object
    # hence we use cls and not self
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

# creates word to number mapping

def build_simple_vocab(
    # collection of tokenized sentences 
    # example: {["a", "man", "runs"], ["a", "dog", "runs"], ["a", "woman", "walks"]}
    token_stream: Iterable[Iterable[str]],
    # only keep words that appear at least 2 times
    min_freq: int = 2,
    # max vocab size
    max_size: Optional[int] = None,
) -> SimpleVocab:
    # counting words
    counter = Counter()
    for tokens in token_stream:
        counter.update(tokens) # count how many times every token appears
    # from above - counter = {"a" : 2, "runs" : 2, "man" : 1, "dog" : 1}

    # Add special tokens first
    itos = list(SPECIAL_TOKENS) # this stats the vocab with ["<unk>",...]

    # fix how many normal words are allowed
    max_extra = None if max_size is None else max(0, max_size - len(itos))

    # loop through the words from most to least frequent
    for token, freq in counter.most_common():
        # if tokens
        if freq < min_freq:
            continue
        if token in SPECIAL_TOKENS:
            continue
        if max_extra is not None and len(itos) - len(SPECIAL_TOKENS) >= max_extra:
            break
        itos.append(token)

    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return SimpleVocab(stoi=stoi, itos=itos)

# Load the Multi30k dataet, tokenize it, convert text to tensors
class Multi30kDataset(Dataset):
    # constructor
    # it inherits from Dataset because PyTorch expects datasets to have 
    # __len__, __getitem__
    # runs when u create the dataset object
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
    # example usage - val_dataset = Multi30kDataset(split="validation", src_vocab=src_vocab, tgt_vocab=tgt_vocab)

     # use valid/val/dev for validation - just for flexibility
        split_map = {
            "valid": "validation",
            "val": "validation",
            "dev": "validation",
        }
        # save the metrics inside the object so any metohod can use them
        self.split = split_map.get(split, split)
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.lower = lower

        # load german and english tokenizers
        self.de_tokenizer = _load_spacy_tokenizer("de")
        self.en_tokenizer = _load_spacy_tokenizer("en")

        # import load_dataset (if datasets package is missing then give errors)
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise ImportError("datasets is required to load Multi30k. Install it with: pip install datasets") from exc

        # this downloads the MUlti30k dataset from Hugging Face
        self.raw_dataset = load_dataset("bentrevett/multi30k", split=self.split)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []

        if self.src_vocab is not None and self.tgt_vocab is not None:
            self.process_data()

    # extact (German sentence, English sentence) from the da
    def _extract_pair(self, example):
        # if direcly as de, eu then use that
        if "de" in example and "en" in example:
            return example["de"], example["en"]
        # if de, eu are present inside a dictionary with key translation
        if "translation" in example:
            tr = example["translation"]
            return tr.get("de", tr.get("deu", "")), tr.get("en", tr.get("eng", ""))
        # named as german, english itself
        if "german" in example and "english" in example:
            return example["german"], example["english"]
        # raise error if key not found
        raise KeyError(f"Could not find German/eng key not found: {list(example.keys())}")

    # returns a list (for german)
    # makes all case lower and tokenizes it word wise
    def _tok_de(self, text: str):
        if self.lower:
            text = text.lower()
        return [tok.text.strip() for tok in self.de_tokenizer.tokenizer(text) if tok.text.strip()]

    # same thing for english
    def _tok_en(self, text: str):
        if self.lower:
            text = text.lower()
        return [tok.text.strip() for tok in self.en_tokenizer.tokenizer(text) if tok.text.strip()]

    # build the vocavb mapping for de to en including the special tokens
    def build_vocab(self):
        src_tokens = []
        tgt_tokens = []
        for ex in self.raw_dataset:
            de, en = self._extract_pair(ex)
            src_tokens.append(self._tok_de(de))
            tgt_tokens.append(self._tok_en(en))

        self.src_vocab = build_simple_vocab(src_tokens, min_freq=self.min_freq, max_size=self.max_vocab_size)
        self.tgt_vocab = build_simple_vocab(tgt_tokens, min_freq=self.min_freq, max_size=self.max_vocab_size)
        return self.src_vocab, self.tgt_vocab

    # convert eng and german sentences to integer token lists usign spacy
    def process_data(self):
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

    # return the number of processed examples
    def __len__(self):
        return len(self.examples)

    # return one example by index (tuple of tensors is returned)
    def __getitem__(self, idx: int):
        return self.examples[idx]

    # combine multiple examples into one batch
    # as sentences have diff length - cannot directly stack them
    # we pad shorter sentences
    def collate_fn(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        # split source and target
        # batch is [(src1, tgt1), (src2, tgt2)]
        src_batch, tgt_batch = zip(*batch)
        # pad source and target
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
        return src_padded, tgt_padded # return the padded sentences
    # as PAD_IDX is 1, we pad with 1
    # shape of src_padded = [batch_size, max_src_len]
    # shape of tgt_padded = [batch_size, max_tgt_len]
