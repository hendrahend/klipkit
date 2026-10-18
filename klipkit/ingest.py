from __future__ import annotations

from pathlib import Path

from .media import extract_audio, media_duration


def resolve_source(source: str, workdir: Path) -> tuple[Path, str, str]:
    workdir.mkdir(parents=True, exist_ok=True)

    local = Path(source).expanduser()
    if local.exists():
        return local, local.stem, "file"

    if source.startswith(("http://", "https://")):
        return _download_url(source, workdir)

    raise FileNotFoundError(f"source not found: {source}")


def _download_url(url: str, workdir: Path) -> tuple[Path, str, str]:
    import yt_dlp

    media_dir = workdir / "source"
    media_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "paths": {"home": str(media_dir)},
        "outtmpl": "%(title).80s [%(id)s].%(ext)s",
        "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/bv*[height<=720]+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "js_runtimes": {"node": {}},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
        if not path.exists():
            candidates = sorted(media_dir.glob("*.mp4"))
            if not candidates:
                raise RuntimeError("download finished but no media file was produced")
            path = candidates[-1]
    return path, info.get("title") or path.stem, "url"


def prepare_audio(media: Path, workdir: Path) -> Path:
    audio = workdir / "audio" / f"{media.stem}.wav"
    if audio.exists() and audio.stat().st_size > 1024:
        return audio
    return extract_audio(media, audio)


__all__ = ["resolve_source", "prepare_audio", "media_duration"]
