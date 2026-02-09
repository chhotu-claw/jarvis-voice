# Setup Details & Troubleshooting

## System Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (for faster-whisper; CPU fallback works but is slow)
- ffmpeg (for webm→PCM audio conversion)
- ~500MB disk for models (Kokoro ONNX + Whisper small.en)
- Ollama running locally with `llama3.1:8b` pulled

## Python Dependencies

Core packages:
```
fastapi
uvicorn[standard]
websockets
httpx
numpy
kokoro-onnx          # TTS — Kokoro v1.0 ONNX runtime
faster-whisper       # STT — CTranslate2-based Whisper
openwakeword         # Wake word detection
edge-tts             # Fallback TTS via Microsoft Edge
```

## Models

| Model | Size | Location |
|-------|------|----------|
| Kokoro v1.0 ONNX | ~310MB | `~/chhotu-voice/kokoro-v1.0.onnx` |
| Kokoro voices | ~27MB | `~/chhotu-voice/voices-v1.0.bin` |
| Whisper small.en | ~460MB | Auto-downloaded by faster-whisper to `~/.cache/` |
| hey_jarvis_v0.1 | ~2MB | Bundled with openwakeword package |

## Ollama Setup

```bash
ollama pull llama3.1:8b
ollama serve  # default port 11434
```

## OpenClaw Gateway

The voice assistant routes computer-control tasks to Claude via the OpenClaw gateway.
Configure `OPENCLAW_GATEWAY` and `OPENCLAW_TOKEN` in `app/server.py`.

## Troubleshooting

**No audio from browser:** Ensure HTTPS or localhost — browsers block mic on plain HTTP.

**Whisper slow:** Check CUDA is available. `nvidia-smi` should show GPU. If not, Whisper falls back to CPU.

**Wake word not triggering:** Lower `WAKE_WORD_THRESHOLD` (e.g., 0.15). Check browser is sending 16kHz mono PCM.

**Kokoro TTS fails:** Falls back to Edge TTS automatically. Check `kokoro-v1.0.onnx` and `voices-v1.0.bin` exist.

**WebSocket disconnects:** Check uvicorn timeout settings. The server uses keepalive pings.
