"""CTC decoding and error-rate metrics (no external dependencies beyond PyTorch)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from .charset import BLANK_INDEX


def greedy_decode(logits: Tensor, lengths: Optional[Tensor] = None) -> list[list[int]]:
    """Best-path CTC decoding: argmax per step, collapse repeats, drop blanks.

    logits: (B,T,C). lengths: optional (B,) valid time steps per item.
    """
    best = logits.argmax(dim=-1).cpu()  # (B,T)
    results: list[list[int]] = []
    for i in range(best.size(0)):
        steps = best[i] if lengths is None else best[i, : int(lengths[i])]
        decoded: list[int] = []
        previous = -1
        for token in steps.tolist():
            if token != previous and token != BLANK_INDEX:
                decoded.append(token)
            previous = token
        results.append(decoded)
    return results


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance (substitutions, insertions and deletions all cost 1)."""
    if not ref:
        return len(hyp)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def _corpus_error_rate(refs: Sequence[Sequence], hyps: Sequence[Sequence]) -> float:
    if len(refs) != len(hyps):
        raise ValueError(f"Got {len(refs)} references but {len(hyps)} hypotheses.")
    total_ref = sum(len(r) for r in refs)
    if total_ref == 0:
        raise ValueError("All references are empty; the error rate is undefined.")
    return sum(edit_distance(r, h) for r, h in zip(refs, hyps)) / total_ref


def cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Character Error Rate over a corpus: total character edits / total reference characters."""
    return _corpus_error_rate(list(references), list(hypotheses))


def wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Word Error Rate over a corpus: total word edits / total reference words."""
    return _corpus_error_rate([r.split() for r in references], [h.split() for h in hypotheses])