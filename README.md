# Klipkit

**Find the clips worth posting — on your own laptop, with no cloud in the loop.**

Klipkit is a local-first clip finder built for freelance clippers and short-form video editors.
Point it at a podcast, a stream VOD or a client's raw footage; it transcribes the audio with
Whisper, finds the moments that actually work as standalone clips, writes a hook line for each,
and exports a client-ready pack (MP4 + `.srt` captions + a notes sheet).

Nothing leaves your machine. No upload, no per-minute API bill, no NDA problem.

```
                                                    
  source ──► whisper (local) ──► candidate moments ──► local LLM judge ──► clip pack
  file/URL     word timestamps     cheap heuristics      open-weight model     mp4 + srt
```

## Why local, and why open

Freelance clippers are paid per clip, so two things kill their margin: **time** and **API cost**.
Cloud clip tools charge per upload-minute, which makes high-volume work unprofitable, and they
require handing a client's footage to a server the client does not control — a non-starter when
there is an NDA.

Klipkit runs everything locally on open pieces:

| Piece | Role | Why open |
| --- | --- | --- |
| **faster-whisper** | transcription with word-level timestamps | runs offline, no per-minute fee, your audio never uploads |
| **Ollama + an open-weight model (Gemma 3 by default)** | judges which moments are clip-worthy and writes hooks | runs on a laptop GPU, swappable, fine-tunable, free to run forever |
| **ffmpeg** | cutting and caption rendering | boring, reliable, already on your machine |
| **yt-dlp** | optional URL import | no vendor gatekeeping your sources |

The marginal cost of the hundredth clip is zero, which is the whole business case for a
freelancer. And because the judge model is swappable, an editor can trade quality for speed
(`qwen3:4b` for a fast pass, a larger Gemma or Qwen when the clip really matters) without
touching the rest of the pipeline.

## Install

```bash
git clone https://github.com/<you>/klipkit.git && cd klipkit
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requirements: `ffmpeg` on `PATH`, and a local [Ollama](https://ollama.com) server with a model:

```bash
ollama pull gemma3:4b      # ~3 GB, fits a 4 GB laptop GPU
ollama serve
klipkit doctor
```

`klipkit doctor` tells you what is missing, including which local models it can see.

## Use

```bash
# find the best moments in a file or URL
klipkit find podcast-ep12.mp4 -n 5
klipkit find "https://youtu.be/..." --lang id -n 8

# render the client-ready pack from the picked clips
klipkit cut out/clips.json

# or do both in one shot
klipkit run client-footage.mp4 -n 6 --burn
```

Output:

```
out/
  transcript.json      word-level transcript (cached, reuse across runs)
  clips.json           selected clips with score, hook, and reason
  pack/
    clips/01-hook-name.mp4
    captions/01-hook-name.srt
    README.md          run sheet: timecodes, scores, hooks, reasons
```

Typical run on an hour of talk audio: a few minutes of transcription on CPU, plus a few
seconds of judging per shortlisted moment. The verbose model never sees raw hours of text —
cheap heuristics pick the shortlist first, exactly because a model drowning in transcript
makes bad calls.

## How the pipeline works

1. **Ingest** — local file, or download via `yt-dlp` (video fetched once, audio extracted to 16 kHz mono WAV).
2. **Transcribe** — `faster-whisper` with `word_timestamps=True`; utterances are reassembled from
   word timings so clip boundaries snap to real speech.
3. **Cheap prefilter** — overlapping candidate windows are scored heuristically (question openers,
   numbers, filler density, whether the cut starts mid-thought). Only a shortlist moves on.
4. **Local judge** — a small batch prompt asks the local model to score hook strength,
   self-containment, payoff and energy (1–5), and to write a short hook line plus a reason.
5. **Selection** — non-overlapping top clips above a score floor.
6. **Pack** — ffmpeg cuts each clip, `.srt` captions are generated from the same word timings,
   and a `README.md` run sheet is written for the client.

Heuristics and the model are deliberately layered: heuristics can throw a candidate away for
free, and the model only ever spends tokens on things that already look plausible.

## Options

| Flag | Meaning |
| --- | --- |
| `-n, --clips` | how many clips to pick (default 5) |
| `--whisper` | Whisper size: `tiny`…`large-v3` (default `small`) |
| `--llm` | local Ollama model (default `gemma3:4b`) |
| `--lang` | spoken language or `auto` |
| `--min-score` | judge score floor (default 0.45) |
| `--burn` | burn captions into the exported video |
| `-o, --output` | output directory (default `out/`) |
| `--force` | re-transcribe even when a cached transcript exists |

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests cover the pure pipeline logic (utterance assembly, candidate windows, scoring,
non-overlap selection, SRT timing) and need no model or GPU.

## Limitations

- Talk-heavy footage is where this shines; B-roll-driven video has no transcript worth judging.
- Auto-reframing to 9:16 is intentionally **not** in v1 — a bad vertical crop is worse than none.
  Klipkit hands the editor timecodes and captions and lets them reframe in CapCut/Premiere.
- Non-Latin scripts and heavy code-switching depend on the Whisper model's coverage.

## License

MIT — see [LICENSE](LICENSE).
