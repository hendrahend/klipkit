from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .media import require_ffmpeg
from .models import ScoredClip, Word, fmt_time


def _srt_timestamp(seconds: float) -> str:
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def words_in_range(words: list[Word], start: float, end: float) -> list[Word]:
    return [w for w in words if w.end > start and w.start < end]


def build_srt(words: list[Word], start: float, end: float, max_chars: int = 42, max_seconds: float = 3.5) -> str:
    lines: list[str] = []
    index = 1
    chunk: list[Word] = []

    def flush() -> None:
        nonlocal index
        if not chunk:
            return
        text = "".join(w.text for w in chunk).strip()
        if not text:
            chunk.clear()
            return
        lines.append(str(index))
        lines.append(f"{_srt_timestamp(chunk[0].start)} --> {_srt_timestamp(chunk[-1].end)}")
        lines.append(text)
        lines.append("")
        index += 1
        chunk.clear()

    for word in words:
        if chunk and (
            len("".join(w.text for w in chunk)) + len(word.text) > max_chars
            or word.end - chunk[0].start > max_seconds
        ):
            flush()
        chunk.append(word)
    flush()
    return "\n".join(lines).strip() + "\n" if lines else ""


def cut_clip(media: Path, clip: ScoredClip, out_path: Path, captions: Path | None = None, burn: bool = False) -> Path:
    require_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.5, clip.end - clip.start)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{clip.start:.3f}", "-t", f"{duration:.3f}", "-i", str(media)]
    if burn and captions and captions.exists():
        escaped = str(captions).replace("\\", "/").replace(":", "\\:")
        cmd += ["-vf", f"subtitles='{escaped}'"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def slugify(text: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_" else " " for c in text.lower()).split()
    return "-".join(cleaned)[:40] or "clip"


def write_pack(
    media: Path,
    clips: list[ScoredClip],
    words: list[Word],
    out_dir: Path,
    title: str,
    render: bool = True,
    burn: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    captions_dir = out_dir / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)

    rows = ["# Clip pack", "", f"Source: {title}", "", "| # | Start | End | Length | Score | Hook |", "| --- | --- | --- | --- | --- | --- |"]

    for position, clip in enumerate(clips, start=1):
        slug = f"{position:02d}-{slugify(clip.hook or clip.text[:30])}"
        srt_path = captions_dir / f"{slug}.srt"
        srt_path.write_text(build_srt(words_in_range(words, clip.start, clip.end), clip.start, clip.end), encoding="utf-8")
        if render:
            cut_clip(media, clip, clips_dir / f"{slug}.mp4", captions=srt_path, burn=burn)
        rows.append(
            f"| {position} | {fmt_time(clip.start)} | {fmt_time(clip.end)} | {clip.end - clip.start:.1f}s | {clip.score:.2f} | {clip.hook or '-'} |"
        )

    rows += ["", "## Notes", ""]
    for position, clip in enumerate(clips, start=1):
        rows.append(f"{position}. **{clip.hook or 'Untitled'}** ({fmt_time(clip.start)}-{fmt_time(clip.end)}) — {clip.reason or 'no note'}")

    readme = out_dir / "README.md"
    readme.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return readme


def save_clips(path: Path, clips: list[ScoredClip], meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "clips": [clip.to_dict() for clip in clips]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_clips(path: Path) -> tuple[list[ScoredClip], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = [ScoredClip(**clip) for clip in data.get("clips", [])]
    return clips, data.get("meta", {})
