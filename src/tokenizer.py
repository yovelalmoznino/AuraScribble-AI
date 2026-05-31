from __future__ import annotations

from pathlib import Path


class CharTokenizer:
    MODE_TOKENS = ("<auto>", "<he>", "<en>", "<math>")
    SPECIAL_SKIP = frozenset({"<blank>", "<pad>", "<bos>", "<eos>", *MODE_TOKENS})

    def __init__(self, vocab_path: str | Path) -> None:
        self.vocab_path = Path(vocab_path)
        if self.vocab_path.exists():
            lines = self.vocab_path.read_text(encoding="utf-8").splitlines()
            self.vocab = [
                line.replace("\n", "")
                for line in lines
                if line.replace("\n", "") != "" or line == " "
            ]
        else:
            self.vocab = ["<blank>", "<pad>", "<bos>", "<eos>"]

        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.blank_id = self.stoi.get("<blank>", 0)
        self.pad_id = self.stoi.get("<pad>", 1)
        self.bos_id = self.stoi.get("<bos>", self.pad_id)
        self.eos_id = self.stoi.get("<eos>", self.pad_id)
        self.auto_id = self.stoi.get("<auto>", -1)
        self.he_id = self.stoi.get("<he>", -1)
        self.en_id = self.stoi.get("<en>", -1)
        self.math_id = self.stoi.get("<math>", -1)

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _contains_hebrew(text: str) -> bool:
        return any("\u0590" <= ch <= "\u05FF" for ch in text)

    @staticmethod
    def _contains_latin(text: str) -> bool:
        return any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)

    @staticmethod
    def _contains_math_markers(text: str) -> bool:
        return "$" in text or "\\" in text

    @classmethod
    def should_reverse_hebrew(cls, text: str, mode: str | None = None) -> bool:
        """Reverse only for pure Hebrew lines (training + pure-hebrew decode)."""
        mode_key = (mode or "auto").lower()
        if mode_key in ("mixed", "english", "math", "text", "correction", "auto"):
            if cls._contains_latin(text) or cls._contains_math_markers(text):
                return False
        if mode_key in ("english", "math"):
            return False
        if not cls._contains_hebrew(text):
            return False
        if cls._contains_latin(text) or cls._contains_math_markers(text):
            return False
        return True

    def mode_prefix_id(self, mode: str | None) -> int | None:
        key = (mode or "auto").lower()
        mapping = {
            "auto": self.auto_id,
            "text": self.auto_id,
            "hebrew": self.he_id,
            "english": self.en_id,
            "math": self.math_id,
            "mixed": self.auto_id,
            "correction": self.auto_id,
        }
        token_id = mapping.get(key, self.auto_id)
        if token_id is None or token_id < 0:
            return None
        return token_id

    def preprocess_text_for_tokens(
        self, text: str, *, rtl_aware: bool = True, mode: str | None = None
    ) -> str:
        if rtl_aware and self.should_reverse_hebrew(text, mode):
            return text[::-1]
        return text

    def encode(
        self,
        text: str,
        rtl_aware: bool = True,
        add_special_tokens: bool = False,
        mode: str | None = None,
        add_mode_prefix: bool = False,
    ) -> list[int]:
        text = self.preprocess_text_for_tokens(text, rtl_aware=rtl_aware, mode=mode)
        ids: list[int] = []
        if add_special_tokens and self.bos_id != self.pad_id:
            ids.append(self.bos_id)
        if add_mode_prefix:
            prefix_id = self.mode_prefix_id(mode)
            if prefix_id is not None:
                ids.append(prefix_id)
        for ch in text:
            idx = self.stoi.get(ch)
            if idx is not None:
                ids.append(idx)
        if add_special_tokens and self.eos_id != self.pad_id:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: list[int],
        rtl_aware: bool = False,
        mode: str | None = None,
    ) -> str:
        out: list[str] = []
        prev: int | None = None
        for idx in ids:
            if idx in (self.blank_id, self.pad_id, self.bos_id, self.eos_id):
                prev = None
                continue
            if idx in (self.auto_id, self.he_id, self.en_id, self.math_id):
                prev = None
                continue
            if prev == idx:
                continue
            prev = idx
            if 0 <= idx < len(self.vocab):
                tok = self.vocab[idx]
                if tok not in self.SPECIAL_SKIP:
                    out.append(tok)
        text = "".join(out)
        if rtl_aware and self.should_reverse_hebrew(text, mode):
            return text[::-1]
        return text
