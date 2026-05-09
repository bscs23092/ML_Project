import json

import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class CharTokenizer:
    def __init__(self, max_len: int = 256):
        self.max_len = max_len
        self.char2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2char = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self._is_fitted = False

    def fit(self, sentences):
        if self._is_fitted:
            raise RuntimeError(
                "CharTokenizer.fit() called more than once. Create a new tokenizer for each fit."
            )

        chars = set()
        for sentence in sentences:
            chars.update(str(sentence))

        for ch in sorted(chars):
            idx = len(self.char2idx)
            self.char2idx[ch] = idx
            self.idx2char[idx] = ch

        self._is_fitted = True
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)

    def encode(self, text: str) -> torch.LongTensor:
        if not self._is_fitted:
            raise RuntimeError("Call fit() on training data before encoding.")

        text = str(text)[: self.max_len]
        ids = [self.char2idx.get(ch, 1) for ch in text]
        ids.extend([0] * (self.max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def transform(self, sentences) -> list[torch.LongTensor]:
        return [self.encode(sentence) for sentence in sentences]

    def save(self, path: str):
        state = {
            "max_len": self.max_len,
            "char2idx": self.char2idx,
            "_is_fitted": self._is_fitted,
        }
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)

        tok = cls(max_len=data["max_len"])
        tok.char2idx = data["char2idx"]
        tok.idx2char = {v: k for k, v in tok.char2idx.items()}
        tok._is_fitted = data["_is_fitted"]
        return tok


class XSSDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer: CharTokenizer):
        if not tokenizer._is_fitted:
            raise RuntimeError("Pass a fitted tokenizer to XSSDataset.")

        self.tokenizer = tokenizer
        self.sentences = list(sentences)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.sentences[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y


def assert_no_leakage(X_train, X_val, X_test, tokenizer: CharTokenizer):
    assert tokenizer._is_fitted, "Tokenizer is not fitted. Call tok.fit(X_train) first."
    assert len(X_train) > 0, "X_train is empty."
    assert len(X_val) > 0, "X_val is empty."
    assert len(X_test) > 0, "X_test is empty."

    train_set = {str(s) for s in X_train}
    val_leak = sum(1 for s in X_val if str(s) in train_set)
    test_leak = sum(1 for s in X_test if str(s) in train_set)

    if val_leak > 0:
        print(f"[LeakageCheck] WARNING: {val_leak} validation samples also appear in train.")
    if test_leak > 0:
        print(f"[LeakageCheck] WARNING: {test_leak} test samples also appear in train.")
