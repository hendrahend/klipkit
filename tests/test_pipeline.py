from pathlib import Path

from klipkit.highlight import Advice, select_clips
from klipkit.models import ScoredClip, Word
from klipkit.render import build_srt, words_in_range
from klipkit.text import build_candidates, build_utterances, heuristic_score, rank_candidates


def make_words(spec):
    words = []
    for text, start, end in spec:
        words.append(Word(start=start, end=end, text=text))
    return words


def test_build_utterances_splits_on_gaps():
    words = make_words(
        [
        ("Halo ", 0.0, 0.5),
        ("kawan. ", 0.5, 1.0),
        ("Ini ", 2.5, 3.0),
        ("topik ", 3.0, 3.4),
        ("baru.", 3.4, 3.9),
        ]
    )
    utterances = build_utterances(words, max_gap=0.7)
    assert len(utterances) == 2
    assert utterances[0].text.startswith("Halo")
    assert utterances[1].start == 2.5


def test_candidates_stay_within_bounds():
    words = []
    for index in range(60):
        start = index * 2.0
        words.append(Word(start=start, end=start + 1.8, text=f"kata{index}."))
    utterances = build_utterances(words)
    candidates = build_candidates(utterances, target_len=40.0)
    assert candidates
    for candidate in candidates:
        assert candidate.end > candidate.start
        assert candidate.end - candidate.start <= 90.0


def test_heuristic_rewards_questions_and_penalises_leading_conjunction():
    good = heuristic_score(_candidate("Kenapa channel kamu tidak tumbuh? Ini 3 cara memperbaikinya."), 40.0)
    weak = heuristic_score(_candidate("Dan terus seperti itu sampai selesai."), 40.0)
    assert good > weak


def test_rank_candidates_keeps_best():
    candidates = [_candidate("biasa saja", index) for index in range(20)]
    candidates[3].heuristic = 5.0
    ranked = rank_candidates(candidates, keep_ratio=0.5, min_keep=4, max_keep=10)
    assert len(ranked) == 10
    assert any(c.index == 3 for c in ranked)


def test_select_clips_drops_overlaps_and_low_scores():
    clips = [
        ScoredClip(start=0, end=30, text="a", score=0.9),
        ScoredClip(start=10, end=40, text="b", score=0.8),
        ScoredClip(start=60, end=90, text="c", score=0.7),
        ScoredClip(start=200, end=230, text="d", score=0.2),
    ]
    chosen = select_clips(clips, Advice(max_clips=5, min_score=0.45))
    assert [c.text for c in chosen] == ["a", "c"]


def test_build_srt_offsets_and_groups():
    words = make_words(
        [
            ("Ini ", 10.0, 10.4),
            ("contoh ", 10.4, 10.9),
            ("kalimat.", 10.9, 11.6),
        ]
    )
    srt = build_srt(words, 10.0, 12.0)
    assert srt.startswith("1\n")
    assert "00:00:10,000 --> 00:00:11,600" in srt


def test_words_in_range_filters():
    words = make_words([("a", 0, 1), ("b", 1, 2), ("c", 5, 6)])
    kept = words_in_range(words, 0.5, 2.5)
    assert [w.text for w in kept] == ["a", "b"]


def _candidate(text, index=0):
    from klipkit.models import Candidate

    return Candidate(index=index, start=0.0, end=40.0, text=text)
