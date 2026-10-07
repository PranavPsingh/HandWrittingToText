"""Character tokenizer for CTC-based handwriting recognition.

Index 0 is always the CTC blank. Real characters start at index 1.
The character set is saved as JSON next to the exported model so the
application loads it from a local file and never has to guess it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

BLANK_INDEX = 0


@dataclass(frozen=True)
class CharTokenizer:
    """Maps characters to integer ids. ``chars[i]`` has id ``i + 1``."""

    chars: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("CharTokenizer.chars contains duplicate characters.")
        if any(len(c) != 1 for c in self.chars):
            raise ValueError("Every entry in CharTokenizer.chars must be a single character.")

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharTokenizer":
        """Build the character set from training transcriptions only (never from test data)."""
        return cls(tuple(sorted(set("".join(texts)))))

    @property
    def num_classes(self) -> int:
        """Number of output classes including the CTC blank."""
        return len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        lookup = {c: i + 1 for i, c in enumerate(self.chars)}
        unknown = sorted({c for c in text if c not in lookup})
        if unknown:
            raise ValueError(
                f"Characters not in the character set: {unknown!r}. "
                "Rebuild the tokenizer from the training transcriptions or clean the text first."
            )
        return [lookup[c] for c in text]

    def decode(self, indices: Sequence[int]) -> str:
        """Turn ids back into text. Blank ids are skipped; repeats are NOT collapsed here."""
        out = []
        for i in indices:
            if i == BLANK_INDEX:
                continue
            if not 0 < i <= len(self.chars):
                raise ValueError(f"Index {i} is outside the valid range 0..{len(self.chars)}.")
            out.append(self.chars[i - 1])
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"blank_index": BLANK_INDEX, "chars": list(self.chars)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("blank_index") != BLANK_INDEX:
            raise ValueError(f"Unsupported blank_index in {path}: {data.get('blank_index')!r}")
        return cls(tuple(data["chars"]))