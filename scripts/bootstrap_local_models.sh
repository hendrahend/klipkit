#!/usr/bin/env bash
# Bootstrap the local stack Klipkit depends on: ffmpeg, a local Ollama server with
# an open-weight Gemma model, and the Whisper weights used for transcription.
#
# Everything here is optional help: Klipkit itself only needs `ffmpeg` on PATH and
# an Ollama server it can reach. This script installs both without root.
set -euo pipefail

ROOT="${KLIPKIT_HOME:-$HOME/.local/opt}"
BIN_DIR="${HOME}/.local/bin"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_VERSION="${OLLAMA_VERSION:-v0.40.5}"
LLM_MODEL="${LLM_MODEL:-gemma3:1b}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"

PYTHON="$PROJECT_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

mkdir -p "$ROOT" "$BIN_DIR"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

install_ollama() {
  if [ -x "$ROOT/ollama/bin/ollama" ]; then
    log "ollama already extracted"
    return
  fi
  local archive="$ROOT/ollama-linux-amd64.tar.zst"
  local url="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
  log "downloading ollama ${OLLAMA_VERSION} to $archive"
  until curl -fL --retry 5 --retry-all-errors -C - -o "$archive" "$url"; do
    log "download interrupted, resuming"
    sleep 3
  done
  mkdir -p "$ROOT/ollama"
  tar --zstd -xf "$archive" -C "$ROOT/ollama"
  ln -sf "$ROOT/ollama/bin/ollama" "$BIN_DIR/ollama"
  log "ollama installed at $ROOT/ollama/bin/ollama"
}

start_ollama() {
  if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    log "ollama server already running"
    return
  fi
  log "starting ollama server in the background"
  nohup "$ROOT/ollama/bin/ollama" serve >"$ROOT/ollama-serve.log" 2>&1 &
  for _ in $(seq 1 30); do
    if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
      log "ollama is up"
      return
    fi
    sleep 2
  done
  log "ollama did not answer in time, see $ROOT/ollama-serve.log"
}

pull_model() {
  log "pulling local model $LLM_MODEL"
  "$ROOT/ollama/bin/ollama" pull "$LLM_MODEL"
}

pull_whisper() {
  log "warming Whisper $WHISPER_MODEL weights (first transcription downloads them)"
  "$PYTHON" - "$WHISPER_MODEL" <<'PY'
import sys
from faster_whisper import WhisperModel

size = sys.argv[1]
WhisperModel(size, device="cpu", compute_type="int8")
print(f"whisper {size} ready")
PY
}

install_ollama
start_ollama
pull_model
pull_whisper

log "done. Run: klipkit doctor"
