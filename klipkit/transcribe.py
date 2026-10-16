from __future__ import annotations

from pathlib import Path

from .media import load_transcript, save_transcript
from .models import Word


def transcribe(
    audio: Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str | None = None,
    language: str | None = None,
    beam_size: int = 1,
    model_dir: Path | None = None,
    on_progress=None,
) -> tuple[list[Word], dict]:
    from faster_whisper import WhisperModel

    if compute_type is None:
        compute_type = "float16" if device in ("cuda", "auto") else "int8"

    try:
        model = WhisperModel(str(model_size), device=device, compute_type=compute_type, download_root=str(model_dir) if model_dir else None)
    except Exception:
        model = WhisperModel(str(model_size), device="cpu", compute_type="int8", download_root=str(model_dir) if model_dir else None)

    segments, info = model.transcribe(
        str(audio),
        language=language,
        word_timestamps=True,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )

    words: list[Word] = []
    total = getattr(info, "duration", 0.0) or 0.0
    for segment in segments:
        if on_progress and total:
            on_progress(segment.end, total)
        for word in segment.words or []:
            text = word.word
            if text.strip():
                words.append(Word(start=float(word.start), end=float(word.end), text=text))

    meta = {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": total,
        "model": str(model_size),
        "device": device,
        "compute_type": compute_type,
    }
    return words, meta


def transcribe_cached(
    audio: Path,
    cache_path: Path,
    force: bool = False,
    **kwargs,
) -> tuple[list[Word], dict]:
    if cache_path.exists() and not force:
        return load_transcript(cache_path)
    words, meta = transcribe(audio, **kwargs)
    save_transcript(cache_path, words, meta)
    return words, meta
