from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .models import Word


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    return path


def extract_audio(media: Path, out_wav: Path) -> Path:
    require_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(media),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    return out_wav


def media_duration(media: Path) -> float:
    require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(media),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    try:
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return 0.0


def load_transcript(path: Path) -> tuple[list[Word], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    words = [Word(**w) for w in data["words"]]
    return words, data.get("meta", {})


def save_transcript(path: Path, words: list[Word], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "words": [w.__dict__ for w in words]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
