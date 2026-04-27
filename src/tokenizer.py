from __future__ import annotations
import json
from pathlib import Path

class CharTokenizer:
    def __init__(self, vocab_path: str | Path) -> None:
        self.vocab_path = Path(vocab_path)
        if self.vocab_path.exists():
            # קוראים את השורות ושומרים על רווחים (לא עושים strip() גורף)
            lines = self.vocab_path.read_text(encoding="utf-8").splitlines()
            # אנחנו מסננים שורות ריקות אבל משאירים שורה שהיא רק רווח (" ")
            self.vocab = [line.replace('\n', '') for line in lines if line.replace('\n', '') != '' or line == ' ']
        else:
            self.vocab = ["<blank>", "<pad>", "<bos>", "<eos>"]
            
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.blank_id = self.stoi.get("<blank>", 0)
        self.pad_id = self.stoi.get("<pad>", 1)
        self.bos_id = self.stoi.get("<bos>", self.pad_id)
        self.eos_id = self.stoi.get("<eos>", self.pad_id)

    def __len__(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self)

    @staticmethod
    def _contains_hebrew(text: str) -> bool:
        return any("\u0590" <= ch <= "\u05FF" for ch in text)

    def preprocess_text_for_tokens(self, text: str, rtl_aware: bool = True) -> str:
        if rtl_aware and self._contains_hebrew(text):
            return text[::-1]
        return text

    def encode(self, text: str, rtl_aware: bool = True, add_special_tokens: bool = False) -> list[int]:
        text = self.preprocess_text_for_tokens(text, rtl_aware=rtl_aware)
        ids: list[int] = []
        if add_special_tokens and self.bos_id != self.pad_id:
            ids.append(self.bos_id)
        for ch in text:
            idx = self.stoi.get(ch)
            if idx is not None:
                ids.append(idx)
        if add_special_tokens and self.eos_id != self.pad_id:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], rtl_aware: bool = False) -> str:
        out = []
        prev = None
        for idx in ids:
            if idx in (self.blank_id, self.pad_id, self.bos_id, self.eos_id):
                prev = None
                continue
            if prev == idx:
                continue
            prev = idx
            if 0 <= idx < len(self.vocab):
                tok = self.vocab[idx]
                if tok not in ("<blank>", "<pad>", "<bos>", "<eos>"):
                    out.append(tok)
        text = "".join(out)
        if rtl_aware and self._contains_hebrew(text):
            return text[::-1]
        return text
