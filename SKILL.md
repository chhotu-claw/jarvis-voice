---
name: jarvis-voice
description: >
  Jarvis voice assistant with 3D face UI — real-time STT (faster-whisper), TTS (Kokoro ONNX),
  wake word detection ("Hey Jarvis" via openwakeword), and LLM routing (local Ollama + OpenClaw gateway).
  Use when setting up, running, or troubleshooting the Jarvis/Chhotu voice assistant server.
  Triggers: voice assistant, jarvis, chhotu voice, speech-to-text, text-to-speech, wake word.
---

# Jarvis Voice Assistant

Real-time voice assistant with a 3D animated face, push-to-talk, and "Hey Jarvis" wake word.

## Architecture

```
Browser (mic/speaker) ↔ WebSocket ↔ FastAPI server
                                      ├── openwakeword (wake word detection)
                                      ├── faster-whisper (STT, CUDA)
                                      ├── Ollama (local LLM)
                                      ├── Kokoro ONNX (TTS)
                                      └── OpenClaw gateway (task routing)
```

## Stack

| Component | Tech | Config |
|-----------|------|--------|
| STT | faster-whisper `small.en` | CUDA, float16 |
| TTS | Kokoro ONNX v1.0 | Voice: `am_santa`, speed: 0.9 |
| Wake word | openwakeword | `hey_jarvis_v0.1.onnx`, threshold: 0.25 |
| LLM | Ollama `llama3.1:8b` | Local, 60s timeout |
| Routing | OpenClaw gateway | Port 18789, for desktop tasks |
| Server | FastAPI + uvicorn | Port 8080 |
| Frontend | Three.js + facecap 3D face | WebSocket at `/ws/voice` |

## Setup

Run the setup script (creates venv, installs deps, downloads models):

```bash
bash scripts/setup.sh
```

See `references/setup-details.md` for manual setup or troubleshooting.

## Running

```bash
cd ~/chhotu-voice
source bin/activate
python app/server.py
# → http://localhost:8080
```

Prerequisites: Ollama running with `llama3.1:8b`, CUDA available for Whisper.

## Key Files

- `app/server.py` — Main server (WebSocket voice pipeline, TTS, STT, LLM, routing)
- `app/static/index.html` — Frontend UI (mic capture, audio playback, wake word streaming)
- `app/static/face3d.js` — Three.js 3D face with lip-sync and expressions
- `app/static/facecap.glb` — 3D face model (ARKit blendshapes)

## WebSocket Protocol

Connect to `ws://host:8080/ws/voice`. Message types:

**Client → Server:**
- `{"type": "audio", "audio": "<base64>", "sampleRate": 16000, "format": "pcm|webm"}` — PTT audio
- `{"type": "wake_enable"}` / `{"type": "wake_disable"}` — Toggle wake word
- `{"type": "wake_audio", "audio": "<base64>"}` — Continuous audio for wake detection
- `{"type": "text", "text": "..."}` — Text input
- `{"type": "ping"}` — Keepalive

**Server → Client:**
- `{"type": "transcript", "text": "..."}` — STT result
- `{"type": "response", "text": "...", "routed": bool}` — LLM response
- `{"type": "audio_chunk", "chunk": N, "audio": "<base64>", "format": "wav"}` — Streamed TTS
- `{"type": "audio_end", "chunks": N}` — TTS complete
- `{"type": "wake_detected"}` — Wake word triggered
- `{"type": "status", "status": "idle|transcribing|thinking|speaking"}` — State updates

## LLM Routing

The local LLM answers questions directly. When a task requires computer control (browser, files, messages), it prefixes the response with `ROUTE:` and the task is forwarded to Claude via the OpenClaw gateway API.

## Configuration

All config is at the top of `app/server.py`:

- `KOKORO_VOICE` / `KOKORO_SPEED` — TTS voice and speed
- `WHISPER_MODEL` — STT model size
- `OLLAMA_MODEL` — Local LLM model
- `WAKE_WORD_THRESHOLD` — Wake sensitivity (0-1, lower = more sensitive)
- `OPENCLAW_GATEWAY` / `OPENCLAW_TOKEN` — Gateway connection
