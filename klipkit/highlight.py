from __future__ import annotations

from dataclasses import dataclass

from .llm import LocalLLM, OllamaError
from .models import Candidate, ScoredClip
from .text import heuristic_score, rank_candidates

SYSTEM_PROMPT = (
    "You are a short-form video editor who cuts clips for TikTok, Reels and Shorts. "
    "You judge transcript excerpts from podcasts and streams. "
    "You always answer with JSON only."
)

BATCH_PROMPT = """Score each candidate excerpt below as a standalone short-form clip.

Criteria (1-5 each):
- hook_strength: does the first sentence stop someone mid-scroll
- self_contained: does it make sense with no outside context
- payoff: is there a clear point, answer, number or punchline
- energy: is the delivery lively rather than a slow tangent

Also write:
- hook: a punchy on-screen title, at most 6 words, same language as the excerpt, no quotes, no hashtags
- reason: one short sentence, same language as the excerpt

Reply with JSON only:
{{"results": [{{"index": <int>, "hook_strength": <int>, "self_contained": <int>, "payoff": <int>, "energy": <int>, "hook": "<string>", "reason": "<string>"}}]}}

Candidates:
{candidates}
"""


@dataclass
class Advice:
    max_clips: int = 5
    min_score: float = 0.45
    overlap_tolerance: float = 2.5
    prefilter_ratio: float = 0.5
    batch_size: int = 4


def _format_candidates(candidates: list[Candidate]) -> str:
    lines = []
    for candidate in candidates:
        duration = candidate.end - candidate.start
        lines.append(
            f"[{candidate.index}] {candidate.start:.1f}s-{candidate.end:.1f}s ({duration:.0f}s): {candidate.text.strip()}"
        )
    return "\n\n".join(lines)


def score_candidates(
    candidates: list[Candidate],
    llm: LocalLLM,
    advice: Advice | None = None,
    on_progress=None,
) -> list[ScoredClip]:
    advice = advice or Advice()
    scored: list[ScoredClip] = []
    batch_size = max(1, advice.batch_size)

    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]
        entries = _score_batch(batch, llm)
        for candidate in batch:
            entry = entries.get(candidate.index)
            if entry is None:
                scored.append(_fallback(candidate))
                continue
            scored.append(_to_scored(candidate, entry))
        if on_progress:
            on_progress(min(offset + batch_size, len(candidates)), len(candidates))

    return scored


def _score_batch(batch: list[Candidate], llm: LocalLLM) -> dict[int, dict]:
    prompt = BATCH_PROMPT.format(candidates=_format_candidates(batch))
    try:
        data = llm.chat_json(SYSTEM_PROMPT, prompt)
    except OllamaError:
        data = {}
    results = data.get("results") if isinstance(data, dict) else data
    entries: dict[int, dict] = {}
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            entries[index] = item
    return entries


def _clamp(value, low: int = 1, high: int = 5) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def _to_scored(candidate: Candidate, entry: dict) -> ScoredClip:
    hook_strength = _clamp(entry.get("hook_strength", 1))
    self_contained = _clamp(entry.get("self_contained", 1))
    payoff = _clamp(entry.get("payoff", 1))
    energy = _clamp(entry.get("energy", 1))
    weights = (1.5, 1.3, 1.2, 0.5)
    total_weight = sum(weights)
    raw = (
        hook_strength * weights[0]
        + self_contained * weights[1]
        + payoff * weights[2]
        + energy * weights[3]
    ) / (5.0 * total_weight)
    score = round(raw, 3)
    return ScoredClip(
        start=candidate.start,
        end=candidate.end,
        text=candidate.text.strip(),
        hook=str(entry.get("hook", "")).strip().strip('"')[:80],
        reason=str(entry.get("reason", "")).strip()[:200],
        hook_strength=hook_strength,
        self_contained=self_contained,
        payoff=payoff,
        energy=energy,
        score=score,
        heuristic=candidate.heuristic,
    )


def _fallback(candidate: Candidate) -> ScoredClip:
    score = round(min(1.0, max(0.0, candidate.heuristic / 6.0)), 3)
    return ScoredClip(
        start=candidate.start,
        end=candidate.end,
        text=candidate.text.strip(),
        hook="",
        reason="scored by heuristics only (model returned no verdict)",
        score=score,
        heuristic=candidate.heuristic,
    )


def select_clips(scored: list[ScoredClip], advice: Advice | None = None) -> list[ScoredClip]:
    advice = advice or Advice()
    ordered = sorted(scored, key=lambda clip: (clip.score, clip.heuristic), reverse=True)
    chosen: list[ScoredClip] = []
    for clip in ordered:
        if clip.score < advice.min_score:
            continue
        if any(_overlaps(clip, kept, advice.overlap_tolerance) for kept in chosen):
            continue
        chosen.append(clip)
        if len(chosen) >= advice.max_clips:
            break
    return sorted(chosen, key=lambda clip: clip.start)


def _overlaps(a: ScoredClip, b: ScoredClip, tolerance: float) -> bool:
    overlap = min(a.end, b.end) - max(a.start, b.start)
    return overlap > tolerance


def analyze_candidates(candidates: list[Candidate], llm: LocalLLM, advice: Advice | None = None, on_progress=None):
    advice = advice or Advice()
    for candidate in candidates:
        candidate.heuristic = heuristic_score(candidate, candidate.end - candidate.start)
    shortlist = rank_candidates(candidates, keep_ratio=advice.prefilter_ratio)
    scored = score_candidates(shortlist, llm, advice, on_progress=on_progress)
    return select_clips(scored, advice), shortlist
