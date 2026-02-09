"""
Chhotu Voice Assistant — Backend Server
Handles: mic audio → Whisper → LLM (local or Claude) → Piper TTS → audio out
Wake word detection: "hey jarvis" via openwakeword (browser streams audio continuously)
"""

import asyncio
import base64
import io
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import yaml

import numpy as np
import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Chhotu Voice Assistant")

# ── Config ──────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
PIPER_MODEL = str(Path.home() / "chhotu-voice/voices/voice.onnx")
MEMORY_DIR = str(Path.home() / ".openclaw/workspace")
WHISPER_MODEL = "small.en"
WAKE_WORD_THRESHOLD = 0.25  # sensitivity (0-1, lower = more sensitive)
EDGE_TTS_VOICE = "en-US-GuyNeural"  # fallback
KOKORO_MODEL = str(Path.home() / "chhotu-voice/kokoro-v1.0.onnx")
KOKORO_VOICES = str(Path.home() / "chhotu-voice/voices-v1.0.bin")
KOKORO_VOICE = "am_santa"  # default voice
KOKORO_SPEED = 0.9

# OpenClaw gateway for routing tasks to Claude
OPENCLAW_GATEWAY = "http://localhost:18789"
OPENCLAW_TOKEN = "1f25ead148b815f60be7a33515ec38f44ff0355fb1c85cff"

# ── Load memory context for local LLM ──────────────────────────────────
def load_memory_context():
    """Load SOUL.md, USER.md, and recent memory for local LLM context."""
    context = ""
    for fname in ["SOUL.md", "USER.md"]:
        fpath = os.path.join(MEMORY_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                context += f"\n--- {fname} ---\n{f.read()}\n"
    
    today = time.strftime("%Y-%m-%d")
    mem_path = os.path.join(MEMORY_DIR, "memory", f"{today}.md")
    if os.path.exists(mem_path):
        with open(mem_path) as f:
            context += f"\n--- Today's Notes ---\n{f.read()}\n"
    
    mem_long = os.path.join(MEMORY_DIR, "MEMORY.md")
    if os.path.exists(mem_long):
        with open(mem_long) as f:
            context += f"\n--- Long-term Memory ---\n{f.read()}\n"
    
    return context

SYSTEM_PROMPT = """You are Chhotu 🐣, Edemon's AI assistant. You're direct, no-fluff, and get stuff done.

You're running as a voice assistant. Keep responses SHORT and conversational — this will be spoken aloud.
2-3 sentences max unless asked for detail.

ANSWER DIRECTLY: questions, conversation, opinions, weather, time, math, explanations, general knowledge, jokes, advice. You know things — use your knowledge. Don't route simple questions.

ONLY use ROUTE: when the user asks you to DO something on the computer that you physically cannot do, like:
- Open a browser, visit a website
- Send a message/email  
- Print something
- Create/edit/search files
- Run code or scripts
- Control the desktop

Format: ROUTE: <brief task description>

NEVER route questions. NEVER route "what is", "how is", "tell me about", etc. Just answer them.

{memory}
"""

# ── Wake Word Model (lazy load) ─────────────────────────────────────────
_wakeword_model = None

def get_wakeword_model():
    global _wakeword_model
    if _wakeword_model is None:
        from openwakeword.model import Model
        import openwakeword
        pkg_dir = os.path.dirname(openwakeword.__file__)
        model_path = os.path.join(pkg_dir, "resources", "models", "hey_jarvis_v0.1.onnx")
        _wakeword_model = Model(wakeword_model_paths=[model_path])
    return _wakeword_model

# ── Whisper (lazy load) ─────────────────────────────────────────────────
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    return _whisper_model

def convert_webm_to_pcm(webm_bytes: bytes) -> bytes:
    """Convert webm/opus audio to 16kHz mono PCM int16 using ffmpeg."""
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as inf:
        inf.write(webm_bytes)
        in_path = inf.name
    out_path = in_path.replace(".webm", ".wav")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", in_path,
            "-ar", "16000", "-ac", "1", "-f", "s16le", out_path
        ], capture_output=True, check=True)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000, audio_format: str = "pcm") -> str:
    """Transcribe audio bytes to text using Whisper."""
    model = get_whisper()
    
    if audio_format == "webm":
        audio_bytes = convert_webm_to_pcm(audio_bytes)
        sample_rate = 16000
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        with wave.open(f, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_bytes)
        temp_path = f.name
    
    try:
        segments, info = model.transcribe(temp_path, language="en")
        text = " ".join(seg.text.strip() for seg in segments)
        return text
    finally:
        os.unlink(temp_path)

# ── TTS (Kokoro ONNX — fast CPU inference) ─────────────────────────────
_kokoro = None

def get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
        print(f"🎙️ Kokoro TTS loaded (voice: {KOKORO_VOICE})")
    return _kokoro

def synthesize_speech(text: str) -> bytes:
    """Full synthesis using Kokoro ONNX."""
    try:
        kokoro = get_kokoro()
        samples, sr = kokoro.create(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
        int16_audio = (samples * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sr)
            wav.writeframes(int16_audio.tobytes())
        return buf.getvalue()
    except Exception as e:
        print(f"Kokoro error: {e}, falling back to Edge TTS")
        import traceback; traceback.print_exc()
        return _edge_tts_fallback(text)

async def synthesize_speech_stream(text: str):
    """Async streaming TTS: yields (chunk_index, wav_bytes) tuples as Kokoro generates each sentence."""
    try:
        kokoro = get_kokoro()
        i = 0
        async for samples, sr in kokoro.create_stream(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
            int16_audio = (samples * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sr)
                wav.writeframes(int16_audio.tobytes())
            yield i, buf.getvalue()
            i += 1
    except Exception as e:
        print(f"Kokoro stream error: {e}")
        import traceback; traceback.print_exc()
        yield 0, _edge_tts_fallback(text)

def _edge_tts_fallback(text: str) -> bytes:
    """Fallback to Edge TTS if XTTS fails."""
    import edge_tts
    import asyncio as _asyncio
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_mp3 = f.name
    temp_wav = temp_mp3.replace(".mp3", ".wav")
    
    try:
        async def _generate():
            communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
            await communicate.save(temp_mp3)
        
        loop = _asyncio.new_event_loop()
        loop.run_until_complete(_generate())
        loop.close()
        
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_mp3, "-ar", "24000", "-ac", "1", temp_wav],
            capture_output=True, timeout=10
        )
        
        with open(temp_wav, "rb") as f:
            return f.read()
    finally:
        for p in [temp_mp3, temp_wav]:
            if os.path.exists(p):
                os.unlink(p)

# ── LLM ─────────────────────────────────────────────────────────────────
conversation_history = []

async def chat_local(user_message: str) -> dict:
    """Chat with local Ollama model. Returns {text, routed}."""
    global conversation_history
    
    memory = load_memory_context()
    system = SYSTEM_PROMPT.format(memory=memory)
    
    conversation_history.append({"role": "user", "content": user_message})
    
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
    
    messages = [{"role": "system", "content": system}] + conversation_history
    
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
        })
        data = resp.json()
        reply = data.get("message", {}).get("content", "Sorry, I couldn't process that.")
    
    conversation_history.append({"role": "assistant", "content": reply})
    
    routed = False
    route_task = None
    if reply.strip().startswith("ROUTE:"):
        routed = True
        route_task = reply.strip()[6:].strip()
        reply = f"On it. Let me handle that."
    
    return {"text": reply, "routed": routed, "route_task": route_task}

async def route_to_claude(task: str) -> str:
    """Send a task to Claude via OpenClaw gateway chat completions API."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{OPENCLAW_GATEWAY}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENCLAW_TOKEN}",
                    "Content-Type": "application/json",
                    "x-openclaw-agent-id": "main",
                },
                json={
                    "model": "openclaw:main",
                    "user": "jarvis-voice",
                    "messages": [{"role": "user", "content": 
                        f"[VOICE COMMAND from Edemon via Jarvis] {task}\n\n"
                        f"Execute this task. You have browser, shell, and all tools available. "
                        f"Respond with a brief spoken summary (1-2 sentences) of what you did. "
                        f"Keep it conversational — this will be read aloud."
                    }]
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                reply = data["choices"][0]["message"]["content"]
                # Trim to something speakable
                if len(reply) > 500:
                    reply = reply[:500] + "... that's the gist of it."
                return reply
            else:
                print(f"❌ OpenClaw route failed: {resp.status_code} {resp.text[:200]}")
                return "I tried but got an error from my backend."
    except Exception as e:
        print(f"❌ Route error: {e}")
        return "Sorry, couldn't reach my backend."

# ── Process voice command (shared logic) ────────────────────────────────
async def stream_tts_to_ws(ws, text: str):
    """Stream TTS audio chunks over websocket as they're generated."""
    chunk_count = 0
    async for i, wav_bytes in synthesize_speech_stream(text):
        await ws.send_json({
            "type": "audio_chunk",
            "chunk": i,
            "audio": base64.b64encode(wav_bytes).decode(),
            "format": "wav",
        })
        chunk_count += 1
    await ws.send_json({"type": "audio_end", "chunks": chunk_count})


async def process_voice(ws: WebSocket, audio_bytes: bytes, sample_rate: int = 16000, audio_format: str = "pcm"):
    """Transcribe → LLM → TTS → send back. Routes to Claude if needed."""
    await ws.send_json({"type": "status", "status": "transcribing"})
    text = await asyncio.to_thread(transcribe_audio, audio_bytes, sample_rate, audio_format)
    
    if not text.strip():
        await ws.send_json({"type": "status", "status": "idle"})
        return
    
    await ws.send_json({"type": "transcript", "text": text})
    await ws.send_json({"type": "status", "status": "thinking"})
    result = await chat_local(text)
    
    await ws.send_json({
        "type": "response",
        "text": result["text"],
        "routed": result["routed"],
        "routeTask": result.get("route_task"),
    })
    
    # Speak the initial response (streamed)
    await ws.send_json({"type": "status", "status": "speaking"})
    await stream_tts_to_ws(ws, result["text"])
    
    # If routed, forward to Claude and speak the result
    if result["routed"] and result.get("route_task"):
        await ws.send_json({"type": "status", "status": "thinking"})
        claude_reply = await route_to_claude(result["route_task"])
        
        # Add Claude's reply to conversation history so Llama has context
        conversation_history.append({"role": "assistant", "content": f"[Backend result]: {claude_reply}"})
        
        await ws.send_json({
            "type": "response",
            "text": claude_reply,
            "routed": True,
        })
        
        await ws.send_json({"type": "status", "status": "speaking"})
        await stream_tts_to_ws(ws, claude_reply)
    
    await ws.send_json({"type": "status", "status": "idle"})

# ── WebSocket endpoint ──────────────────────────────────────────────────
@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    print("🐣 Voice client connected")
    
    # Per-connection wake word model state
    ww_model = None
    wake_word_enabled = False
    wake_counter = [0]
    
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")
            print(f"📨 Received: {msg_type}")
            
            if msg_type == "audio":
                # Push-to-talk: receive audio, transcribe, respond
                audio_b64 = data.get("audio", "")
                sample_rate = data.get("sampleRate", 16000)
                audio_format = data.get("format", "pcm")
                audio_bytes = base64.b64decode(audio_b64)
                await process_voice(ws, audio_bytes, sample_rate, audio_format)
            
            elif msg_type == "wake_audio":
                # Continuous audio stream for wake word detection
                if not wake_word_enabled:
                    continue
                
                try:
                    audio_b64 = data.get("audio", "")
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Log audio level periodically
                    wake_counter[0] += 1
                    if wake_counter[0] % 50 == 0:
                        rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2))
                        print(f"🎙️ Audio RMS: {rms:.1f}, samples: {len(audio_int16)}, min/max: {audio_int16.min()}/{audio_int16.max()}")
                    
                    if ww_model is None:
                        ww_model = await asyncio.to_thread(get_wakeword_model)
                    
                    for i in range(0, len(audio_int16) - 1279, 1280):
                        chunk = audio_int16[i:i+1280]
                        prediction = ww_model.predict(chunk)
                        
                        for key, score in prediction.items():
                            if score > 0.01 or wake_counter[0] % 100 == 0:
                                print(f"🔊 {key}: {score:.4f}")
                            
                            if score > WAKE_WORD_THRESHOLD:
                                print(f"🎯 Wake word detected! ({key}: {score:.2f})")
                                ww_model.reset()
                                await ws.send_json({"type": "wake_detected"})
                                break
                except Exception as e:
                    print(f"❌ Wake audio error: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif msg_type == "wake_enable":
                wake_word_enabled = True
                # Lazy-load model
                if ww_model is None:
                    await ws.send_json({"type": "status", "status": "Loading wake word model..."})
                    ww_model = await asyncio.to_thread(get_wakeword_model)
                await ws.send_json({"type": "wake_status", "enabled": True})
                print("🎯 Wake word detection ENABLED")
            
            elif msg_type == "wake_disable":
                wake_word_enabled = False
                await ws.send_json({"type": "wake_status", "enabled": False})
                print("🎯 Wake word detection DISABLED")
            
            elif msg_type == "text":
                text = data.get("text", "")
                if not text.strip():
                    continue
                
                await ws.send_json({"type": "transcript", "text": text})
                await ws.send_json({"type": "status", "status": "thinking"})
                
                result = await chat_local(text)
                
                await ws.send_json({
                    "type": "response",
                    "text": result["text"],
                    "routed": result["routed"],
                    "routeTask": result.get("route_task"),
                })
                
                await ws.send_json({"type": "status", "status": "speaking"})
                await stream_tts_to_ws(ws, result["text"])
                
                await ws.send_json({"type": "status", "status": "idle"})
            
            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        print("🐣 Voice client disconnected")

# ── Service Management ───────────────────────────────────────────────────
def load_services():
    with open(os.path.expanduser("~/service-manager/services.yaml")) as f:
        return yaml.safe_load(f).get("services", {})

def get_systemd_status(unit):
    try:
        result = subprocess.run(["systemctl", "--user", "is-active", unit],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except:
        return "unknown"

def get_service_logs(unit, lines=100):
    try:
        result = subprocess.run(
            ["journalctl", "--user", "-u", unit, "-n", str(lines), "--no-pager"],
            capture_output=True, text=True, timeout=5)
        return result.stdout
    except:
        return "Failed to get logs"

def service_action(unit, action):
    cmd = {"start": "start", "stop": "stop", "restart": "restart"}[action]
    subprocess.run(["systemctl", "--user", cmd, unit], timeout=10)

@app.get("/api/services")
async def list_services():
    services = load_services()
    result = {}
    for sid, svc in services.items():
        status = get_systemd_status(svc["systemd"]) if svc.get("systemd") else "manual"
        result[sid] = {**svc, "id": sid, "systemdStatus": status}
    return result

@app.post("/api/services/{service_id}/{action}")
async def control_service(service_id: str, action: str):
    services = load_services()
    svc = services.get(service_id)
    if not svc or not svc.get("systemd"):
        return {"error": "Not found or no systemd unit"}
    if action not in ("start", "stop", "restart"):
        return {"error": "Invalid action"}
    await asyncio.to_thread(service_action, svc["systemd"], action)
    await asyncio.sleep(1)
    new_status = await asyncio.to_thread(get_systemd_status, svc["systemd"])
    return {"ok": True, "status": new_status}

@app.get("/api/services/{service_id}/logs")
async def service_logs(service_id: str, lines: int = 100):
    services = load_services()
    svc = services.get(service_id)
    if not svc or not svc.get("systemd"):
        return {"error": "No systemd unit"}
    logs = await asyncio.to_thread(get_service_logs, svc["systemd"], lines)
    return {"logs": logs}

@app.get("/api/tunnel")
async def tunnel_status():
    try:
        status = subprocess.run(["systemctl", "--user", "is-active", "cloudflare-tunnel.service"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        url = open("/tmp/tunnel-url.txt").read().strip() if os.path.exists("/tmp/tunnel-url.txt") else None
        return {"status": status, "url": url}
    except:
        return {"status": "unknown", "url": None}

# ── Health check ────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ollama": OLLAMA_MODEL,
        "whisper": WHISPER_MODEL,
        "piper": os.path.exists(PIPER_MODEL),
    }

# ── VNC Proxy ───────────────────────────────────────────────────────────
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.websockets import WebSocket as StarletteWebSocket
import websockets as ws_lib

NOVNC_URL = "http://localhost:6080"

@app.api_route("/vnc/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
async def vnc_proxy(request: Request, path: str):
    """Reverse proxy HTTP requests to noVNC."""
    url = f"{NOVNC_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            request.method, url,
            headers={k: v for k, v in request.headers.items() if k.lower() not in ('host',)},
            content=await request.body(),
        )
    return Response(content=resp.content, status_code=resp.status_code,
                    headers={k: v for k, v in resp.headers.items() if k.lower() not in ('transfer-encoding', 'content-encoding')})

async def _proxy_vnc_ws(ws: WebSocket):
    """Proxy WebSocket connections to noVNC/websockify."""
    # Accept whatever subprotocol the client requests
    print(f"VNC WS: client headers: {dict(ws.headers)}")
    await ws.accept()
    print(f"VNC WS: accepted client, connecting to upstream...")
    try:
        async with ws_lib.connect(
            "ws://localhost:6080/websockify",
            subprotocols=["binary"],
            max_size=None,
            open_timeout=10,
        ) as upstream:
            print(f"VNC WS: upstream connected")
            async def client_to_upstream():
                try:
                    while True:
                        data = await ws.receive_bytes()
                        await upstream.send(data)
                except Exception:
                    pass
            async def upstream_to_client():
                try:
                    async for msg in upstream:
                        if isinstance(msg, bytes):
                            await ws.send_bytes(msg)
                        else:
                            await ws.send_text(msg)
                except Exception:
                    pass
            await asyncio.gather(client_to_upstream(), upstream_to_client())
    except Exception as e:
        import traceback
        print(f"VNC WS proxy error: {e}")
        traceback.print_exc()

@app.websocket("/vnc/websockify")
async def vnc_ws_proxy(ws: WebSocket):
    await _proxy_vnc_ws(ws)

@app.websocket("/websockify")
async def vnc_ws_proxy_root(ws: WebSocket):
    await _proxy_vnc_ws(ws)

# ── Gateway WS Proxy ────────────────────────────────────────────────────
@app.websocket("/ws/gateway")
async def gateway_ws_proxy(client: WebSocket):
    """Proxy WebSocket to OpenClaw gateway on localhost:18789."""
    await client.accept()
    try:
        async with websockets.connect(
            "ws://127.0.0.1:18789",
            origin="http://127.0.0.1:18789",
            additional_headers={"Host": "127.0.0.1:18789"}
        ) as gw:
            async def client_to_gw():
                try:
                    while True:
                        data = await client.receive_text()
                        await gw.send(data)
                except WebSocketDisconnect:
                    pass

            async def gw_to_client():
                try:
                    async for msg in gw:
                        await client.send_text(msg)
                except Exception:
                    pass

            await asyncio.gather(client_to_gw(), gw_to_client())
    except Exception as e:
        print(f"Gateway proxy error: {e}")
    finally:
        try:
            await client.close()
        except:
            pass

# ── Serve frontend ──────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
