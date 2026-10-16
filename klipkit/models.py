from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Candidate:
    index: int
    start: float
    end: float
    text: str
    heuristic: float = 0.0


@dataclass
class ScoredClip:
    start: float
    end: float
    text: str
    hook: str = ""
    reason: str = ""
    hook_strength: int = 0
    self_contained: int = 0
    payoff: int = 0
    energy: int = 0
    score: float = 0.0
    heuristic: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
