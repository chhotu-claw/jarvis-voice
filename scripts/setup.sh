#!/usr/bin/env bash
# Jarvis Voice Assistant — Setup Script
# Creates venv, installs dependencies, downloads models
set -euo pipefail

VOICE_DIR="${VOICE_DIR:-$HOME/chhotu-voice}"
cd "$VOICE_DIR"

echo "🐣 Setting up Jarvis Voice Assistant..."

# ── Python venv ─────────────────────────────────────────────────────────
if [ ! -f bin/activate ]; then
    echo "📦 Creating Python venv..."
    python3 -m venv .
fi
source bin/activate

# ── Dependencies ────────────────────────────────────────────────────────
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install \
    fastapi \
    uvicorn[standard] \
    websockets \
    httpx \
    numpy \
    kokoro-onnx \
    faster-whisper \
    openwakeword \
    edge-tts

# ── Kokoro TTS models ──────────────────────────────────────────────────
KOKORO_MODEL="$VOICE_DIR/kokoro-v1.0.onnx"
KOKORO_VOICES="$VOICE_DIR/voices-v1.0.bin"

if [ ! -f "$KOKORO_MODEL" ]; then
    echo "🎙️ Downloading Kokoro ONNX model (~310MB)..."
    pip install huggingface-hub
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('hexgrad/Kokoro-82M-v1.0-ONNX', 'kokoro-v1.0.onnx', local_dir='$VOICE_DIR')
"
fi

if [ ! -f "$KOKORO_VOICES" ]; then
    echo "🎙️ Downloading Kokoro voices..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('hexgrad/Kokoro-82M-v1.0-ONNX', 'voices-v1.0.bin', local_dir='$VOICE_DIR')
"
fi

# ── Verify CUDA for Whisper ─────────────────────────────────────────────
echo "🔍 Checking CUDA availability..."
python3 -c "import ctranslate2; print(f'CTranslate2 CUDA: {\"cuda\" in ctranslate2.get_supported_compute_types(\"cuda\")}')" 2>/dev/null || echo "⚠️  CUDA not available — Whisper will use CPU (slower)"

# ── Pre-download Whisper model ──────────────────────────────────────────
echo "🎤 Pre-downloading Whisper small.en model..."
python3 -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='auto')" 2>/dev/null || true

# ── Check ffmpeg ────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo "⚠️  ffmpeg not found — install it: sudo apt install ffmpeg"
fi

echo ""
echo "✅ Setup complete!"
echo "   Start: cd $VOICE_DIR && source bin/activate && python app/server.py"
echo "   Open:  http://localhost:8080"
