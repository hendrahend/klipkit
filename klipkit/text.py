from __future__ import annotations

import re

from .models import Candidate, Utterance, Word

SENTENCE_END = re.compile(r"[.!?。！？][\"')\]]?$")
FILLERS = {"eh", "emm", "hmm", "uh", "um", "anu", "kayak", "gitu", "jadi", "oke", "ya"}


def build_utterances(words: list[Word], max_gap: float = 0.7, max_len: float = 14.0) -> list[Utterance]:
    utterances: list[Utterance] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        text = "".join(w.text for w in current).strip()
        if text:
            utterances.append(
                Utterance(
                    start=current[0].start,
                    end=current[-1].end,
                    text=text,
                    words=list(current),
                )
            )
        current.clear()

    for word in words:
        stripped = word.text.strip()
        if not stripped:
            continue
        if not current:
            current.append(word)
            continue
        gap = word.start - current[-1].end
        too_long = word.end - current[0].start > max_len
        if gap > max_gap or too_long:
            flush()
            current.append(word)
            continue
        current.append(word)
        if SENTENCE_END.search(stripped) and current[-1].end - current[0].start > 1.2:
            flush()

    flush()
    return utterances


def join_text(utterances: list[Utterance]) -> str:
    return " ".join(u.text for u in utterances).strip()


def build_candidates(
    utterances: list[Utterance],
    target_len: float = 45.0,
    min_len: float = 12.0,
    max_len: float = 90.0,
    stride: int = 2,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    total = len(utterances)
    i = 0
    while i < total:
        window: list[Utterance] = []
        j = i
        while j < total:
            window.append(utterances[j])
            span = utterances[j].end - utterances[i].start
            if span >= target_len:
                break
            j += 1
        span = window[-1].end - window[0].start if window else 0.0
        if span >= min_len:
            candidates.append(
                Candidate(
                    index=len(candidates),
                    start=window[0].start,
                    end=min(window[-1].end, window[0].start + max_len),
                    text=join_text(window),
                )
            )
        i += max(1, stride)
    return candidates


def heuristic_score(candidate: Candidate, duration: float) -> float:
    text = candidate.text.strip()
    if not text:
        return 0.0
    words = text.split()
    n = len(words)
    if n < 8:
        return 0.0

    score = 0.0
    lower = text.lower()

    if "?" in text:
        score += 1.5
    if any(c.isdigit() for c in text):
        score += 0.8
    if any(marker in lower for marker in ("cara", "tips", "rahasia", "gratis", "hasilnya", "kenapa", "ternyata", "how to", "secret", "free")):
        score += 1.2
    if lower.startswith(("dan ", "tapi ", "so ", "jadi ", "karena ")):
        score -= 1.5
    if text.endswith((",", "dan", "tapi", "karena", "yang", "di", "ke")):
        score -= 1.0

    fillers = sum(1 for w in words if w.strip(",.!?").lower() in FILLERS)
    score -= 1.5 * (fillers / n)

    score += min(1.0, n / 120.0)
    if 25.0 <= duration <= 75.0:
        score += 1.0
    elif duration < 16.0 or duration > 95.0:
        score -= 1.0

    return round(score, 3)


def rank_candidates(candidates: list[Candidate], keep_ratio: float = 0.5, min_keep: int = 8, max_keep: int = 16) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda c: c.heuristic, reverse=True)
    keep = max(min_keep, int(len(candidates) * keep_ratio))
    keep = min(keep, max_keep, len(ranked))
    selected = ranked[:keep]
    return sorted(selected, key=lambda c: c.start)
