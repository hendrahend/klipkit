from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .highlight import Advice, analyze_candidates
from .ingest import prepare_audio, resolve_source
from .llm import DEFAULT_MODEL, LocalLLM, OllamaError
from .media import media_duration
from .models import fmt_time
from .render import load_clips, save_clips, write_pack
from .text import build_candidates, build_utterances
from .transcribe import transcribe_cached

app = typer.Typer(add_completion=False, help="Klipkit - local-first clip finder for freelance clippers.")
console = Console()


def _workdir(output: str) -> Path:
    return Path(output).expanduser()


@app.command()
def doctor(
    model: str = typer.Option(DEFAULT_MODEL, help="Local model used for clip scoring."),
    whisper_model: str = typer.Option("small", help="Whisper model size for transcription."),
) -> None:
    """Check that ffmpeg, whisper and the local model are ready."""
    table = Table(title="Klipkit doctor")
    table.add_column("Check")
    table.add_column("Status")

    import shutil

    table.add_row("ffmpeg", "[green]ok[/green]" if shutil.which("ffmpeg") else "[red]missing[/red]")
    try:
        import faster_whisper  # noqa: F401

        table.add_row("faster-whisper", "[green]ok[/green]")
    except Exception:
        table.add_row("faster-whisper", "[red]missing[/red]")

    client = LocalLLM(model=model)
    version = client.version()
    if version:
        table.add_row("ollama server", f"[green]ok[/green] ({version})")
        try:
            installed = client.models()
            table.add_row("local models", ", ".join(installed) or "[yellow]none[/yellow]")
            client.ensure_model()
            table.add_row(f"model {model}", "[green]ok[/green]")
        except OllamaError as exc:
            table.add_row(f"model {model}", f"[red]{exc}[/red]")
    else:
        table.add_row("ollama server", "[red]not reachable on 127.0.0.1:11434[/red]")

    console.print(table)


@app.command()
def find(
    source: str = typer.Argument(..., help="Video file or YouTube/URL source."),
    clips: int = typer.Option(5, "-n", "--clips", help="How many clips to pick."),
    whisper_model: str = typer.Option("small", "--whisper", help="Whisper model size."),
    model: str = typer.Option(DEFAULT_MODEL, "--llm", help="Local model used for scoring."),
    language: str = typer.Option("auto", "--lang", help="Spoken language, or auto."),
    output: str = typer.Option("out", "-o", "--output", help="Working/output directory."),
    min_score: float = typer.Option(0.45, help="Minimum judge score to keep a clip."),
    force: bool = typer.Option(False, "--force", help="Ignore cached transcript."),
) -> None:
    """Transcribe a source and find the best clip moments."""
    workdir = _workdir(output)
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("resolving source", total=None)
        media, title, kind = resolve_source(source, workdir)
        progress.update(task, description=f"source ready ({kind}): {title}")

        audio = prepare_audio(media, workdir)
        duration = media_duration(media)
        progress.update(task, description=f"transcribing with whisper:{whisper_model} ({duration/60:.1f} min audio)")

        def on_transcribe(end: float, total: float) -> None:
            progress.update(task, description=f"transcribing... {end/max(total,1e-6)*100:.0f}%")

        words, meta = transcribe_cached(
            audio,
            workdir / "transcript.json",
            force=force,
            model_size=whisper_model,
            language=None if language == "auto" else language,
            on_progress=on_transcribe,
        )

        utterances = build_utterances(words)
        candidates = build_candidates(utterances)
        progress.update(task, description=f"scoring {len(candidates)} candidate moments with {model}")

        def on_score(done: int, total: int) -> None:
            progress.update(task, description=f"scoring moments... {done}/{total}")

        llm = LocalLLM(model=model)
        llm.ensure_model()
        advice = Advice(max_clips=clips, min_score=min_score)
        try:
            selected, shortlist = analyze_candidates(candidates, llm, advice, on_progress=on_score)
        except OllamaError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

        save_clips(
            workdir / "clips.json",
            selected,
            meta={**meta, "title": title, "source": str(media), "candidates": len(candidates), "shortlist": len(shortlist)},
        )
        progress.update(task, description="done")

    _print_clips(selected, title)
    console.print(f"\nSaved [bold]{workdir / 'clips.json'}[/bold]. Render with: [cyan]klipkit cut {workdir / 'clips.json'}[/cyan]")


@app.command()
def cut(
    clips_json: str = typer.Argument(..., help="Path to clips.json produced by `klipkit find`."),
    output: str = typer.Option(None, "-o", "--output", help="Output directory (defaults to beside clips.json)."),
    burn: bool = typer.Option(False, "--burn", help="Burn captions into the video."),
) -> None:
    """Render clips and caption files from a clips.json."""
    path = Path(clips_json).expanduser()
    clips, meta = load_clips(path)
    if not clips:
        console.print("[yellow]No clips in that file.[/yellow]")
        raise typer.Exit(code=0)

    source = Path(meta.get("source", ""))
    if not source.exists():
        console.print(f"[red]Source media not found: {source}[/red]")
        raise typer.Exit(code=1)

    transcript = path.parent / "transcript.json"
    from .media import load_transcript

    words, _ = load_transcript(transcript) if transcript.exists() else ([], {})

    out_dir = Path(output).expanduser() if output else path.parent / "pack"
    with console.status(f"rendering {len(clips)} clips with ffmpeg"):
        write_pack(source, clips, words, out_dir, meta.get("title", source.stem), render=True, burn=burn)

    console.print(f"[green]Pack written to {out_dir}[/green]")


@app.command()
def run(
    source: str = typer.Argument(..., help="Video file or YouTube/URL source."),
    clips: int = typer.Option(5, "-n", "--clips"),
    whisper_model: str = typer.Option("small", "--whisper"),
    model: str = typer.Option(DEFAULT_MODEL, "--llm"),
    language: str = typer.Option("auto", "--lang"),
    output: str = typer.Option("out", "-o", "--output"),
    burn: bool = typer.Option(False, "--burn"),
    min_score: float = typer.Option(0.45),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Find clips and render the client-ready pack in one go."""
    find(source, clips=clips, whisper_model=whisper_model, model=model, language=language, output=output, min_score=min_score, force=force)

    workdir = _workdir(output)
    clips_path = workdir / "clips.json"
    from .media import load_transcript

    selected, meta = load_clips(clips_path)
    source_media = Path(meta.get("source", ""))
    words, _ = load_transcript(workdir / "transcript.json")
    pack = workdir / "pack"
    write_pack(source_media, selected, words, pack, meta.get("title", source_media.stem), render=True, burn=burn)
    console.print(f"[green]Client-ready pack: {pack}[/green]")


def _print_clips(clips, title: str) -> None:
    table = Table(title=f"Top moments - {title}")
    table.add_column("#", justify="right")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Len")
    table.add_column("Score", justify="right")
    table.add_column("Hook")
    for position, clip in enumerate(clips, start=1):
        table.add_row(
            str(position),
            fmt_time(clip.start),
            fmt_time(clip.end),
            f"{clip.end - clip.start:.0f}s",
            f"{clip.score:.2f}",
            clip.hook or "-",
        )
    console.print(table)
    for position, clip in enumerate(clips, start=1):
        if clip.reason:
            console.print(f"[bold]{position}.[/bold] {clip.reason}")


@app.command()
def version() -> None:
    """Print the Klipkit version."""
    console.print(Panel(f"klipkit {__version__}", expand=False))


if __name__ == "__main__":
    app()
