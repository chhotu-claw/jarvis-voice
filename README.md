# Jarvis Voice

Real-time voice assistant with a 3D animated face, push-to-talk, and "Hey Jarvis" wake word detection.

## Architecture

```
Browser (mic/speaker) ↔ WebSocket ↔ FastAPI server
                                      ├── openwakeword  (wake word detection)
                                      ├── faster-whisper (STT, CUDA)
                                      ├── Ollama         (local LLM)
                                      ├── Kokoro ONNX    (TTS)
                                      └── OpenClaw       (task routing)
```

## Tech Stack

- **FastAPI + uvicorn** — WebSocket server (port 8080)
- **faster-whisper** — Speech-to-text (`small.en`, CUDA float16)
- **Kokoro ONNX** — Text-to-speech (voice: `am_santa`, speed: 0.9)
- **openwakeword** — Wake word detection (`hey_jarvis_v0.1.onnx`)
- **Ollama** — Local LLM (`llama3.1:8b`)
- **Three.js** — 3D face UI with lip-sync and expressions
- **OpenClaw gateway** — Routes desktop tasks to Claude

## Setup

```bash
# Automated setup (creates venv, installs deps, downloads models)
bash scripts/setup.sh
```

**Prerequisites:** NVIDIA GPU with CUDA, Ollama running with `llama3.1:8b`.

See `references/setup-details.md` for manual setup or troubleshooting.

## Usage

```bash
cd ~/chhotu-voice
source bin/activate
python app/server.py
# → http://localhost:8080
```

Open the URL in a browser to access the 3D face UI. Use push-to-talk or enable "Hey Jarvis" wake word.

## Folder Structure

```
jarvis-voice/
├── app/
│   └── server.py            # Main server — WebSocket voice pipeline, STT, TTS, LLM
├── scripts/
│   └── setup.sh             # Automated setup script
├── references/
│   └── setup-details.md     # Manual setup & troubleshooting
├── SKILL.md                 # OpenClaw agent skill definition
└── COPYING                  # License
```

The frontend (HTML, JS, 3D model) is served from `app/static/` within the deployed environment.

## Configuration

All config lives at the top of `app/server.py`:

| Setting | Description |
|---|---|
| `KOKORO_VOICE` / `KOKORO_SPEED` | TTS voice and speed |
| `WHISPER_MODEL` | STT model size |
| `OLLAMA_MODEL` | Local LLM model |
| `WAKE_WORD_THRESHOLD` | Wake sensitivity (0–1, lower = more sensitive) |
| `OPENCLAW_GATEWAY` / `OPENCLAW_TOKEN` | Gateway connection for task routing |

## LLM Routing

The local LLM handles questions directly. For tasks requiring computer control (browser, files, messages), it prefixes with `ROUTE:` and the request is forwarded to Claude via OpenClaw gateway.
