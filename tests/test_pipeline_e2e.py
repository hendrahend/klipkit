from klipkit.highlight import Advice, analyze_candidates
from klipkit.highlight import score_candidates
from klipkit.models import Candidate
from klipkit.text import build_candidates, build_utterances, heuristic_score
from klipkit.models import Word


class FakeLLM:
    def __init__(self, hook="Rahasianya di detik pertama"):
        self.calls = 0
        self.hook = hook

    def chat_json(self, system, user):
        self.calls += 1
        import re

        indexes = [int(m) for m in re.findall(r"^\[(\d+)\]", user, flags=re.MULTILINE)]
        return {
            "results": [
                {
                    "index": index,
                    "hook_strength": 4,
                    "self_contained": 4,
                    "payoff": 3,
                    "energy": 3,
                    "hook": self.hook,
                    "reason": "starts with a question and answers it",
                }
                for index in indexes
            ]
        }


def _transcript(seconds=600):
    words = []
    step = 0.4
    text = "Ini cerita panjang tentang cara membuat klip yang bagus."
    t = 0.0
    i = 0
    while t < seconds:
        words.append(Word(start=t, end=t + step, text=text[i % len(text)]))
        t += step
        i += 1
    return words


def test_pipeline_produces_non_overlapping_clips_with_hooks():
    words = _transcript()
    utterances = build_utterances(words)
    candidates = build_candidates(utterances)
    for candidate in candidates:
        candidate.heuristic = heuristic_score(candidate, candidate.end - candidate.start)
    llm = FakeLLM()
    selected, shortlist = analyze_candidates(candidates, llm, Advice(max_clips=3, min_score=0.3))

    assert shortlist
    assert selected
    assert llm.calls >= 1
    for position in range(1, len(selected)):
        assert selected[position].start >= selected[position - 1].end - 2.5
    assert all(clip.hook for clip in selected)
    assert all(0 < clip.score <= 1 for clip in selected)


def test_score_candidates_falls_back_when_model_returns_nothing():
    candidates = [Candidate(index=0, start=0, end=30, text="Dan seperti biasa saja tidak ada yang menarik.", heuristic=0.1)]
    clips = score_candidates(candidates, FakeLLM())

    class Broken:
        def chat_json(self, system, user):
            return {}

    clips = score_candidates(candidates, Broken())
    assert len(clips) == 1
    assert clips[0].hook == ""
    assert "heuristics" in clips[0].reason
