# Android Hermes Assistant via Home Assistant — Setup Plan

## Target architecture
Android Home Assistant Companion App provides:
- default assistant app
- lock-screen assistant entry
- experimental wake word detection
- STT/TTS voice pipeline

Home Assistant routes conversation to:
- `WolframRavenwolf/hermes-ha-integration`
- Hermes Agent API server `/v1/chat/completions`

Hermes handles:
- reasoning
- Home Assistant / server / tool actions
- multi-turn response text

## Prerequisites
- Home Assistant 2024.12+
- Android Home Assistant Companion App 2026.2.3+
- HACS installed in Home Assistant
- A running Hermes Agent API endpoint, either:
  1. Hermes Agent Home Assistant add-on (`WolframRavenwolf/hermes-ha-addon`) with API enabled, or
  2. standalone Hermes Agent with API server enabled and reachable by Home Assistant

## Recommended path
Use the Home Assistant add-on first if possible. It keeps network/auth/HA-token wiring simpler than exposing the existing Hermes gateway.

## Phase 1 — Install/run Hermes for Home Assistant

### Option A: Hermes Agent HA add-on
1. In Home Assistant, go to **Settings → Apps → Install app → ⋮ → Repositories**.
2. Add repository:
   `https://github.com/WolframRavenwolf/hermes-ha-addon`
3. Install **Hermes Agent**.
4. In add-on configuration:
   - enable API server
   - set an access password
   - set `homeassistant_token` to a Home Assistant long-lived access token
   - configure model/API env vars as needed
5. Start the add-on.
6. Verify health:
   - `https://homeassistant.local:8443/health`
   - API base should be `https://homeassistant.local:8443/v1/`

### Option B: Existing standalone Hermes
1. Enable Hermes API server.
2. Make sure Home Assistant can reach it over LAN/Tailscale.
3. Record:
   - host
   - port
   - http/https
   - API key / bearer token
4. Confirm `/v1/chat/completions` works before wiring HA.

## Phase 2 — Install Hermes HA conversation integration
1. Open HACS.
2. Open **Custom repositories**.
3. Add:
   `https://github.com/WolframRavenwolf/hermes-ha-integration`
4. Category: **Integration**.
5. Install **Hermes Agent**.
6. Restart Home Assistant.

## Phase 3 — Configure the Hermes Agent integration
1. Go to **Settings → Devices & Services → Add Integration**.
2. Search for **Hermes Agent**.
3. Enter:
   - Host: `homeassistant.local` if using add-on, or the standalone Hermes host
   - Port: default `8443` for add-on HTTPS
   - API key: the add-on access password / standalone API key
   - Use HTTPS: enabled for add-on
   - Verify SSL: disabled if using self-signed cert
4. Submit.
5. Optional: configure entity exposure/system prompt only after basic voice works.

## Phase 4 — Make Hermes the HA voice assistant brain
1. Go to **Settings → Voice Assistants**.
2. Create or edit an assistant.
3. Select **Hermes Agent** as the **Conversation agent**.
4. Disable **Prefer handling commands locally** so Hermes handles the request.
5. Pick a working STT/TTS pipeline:
   - easiest/fastest: Home Assistant Cloud voice pipeline
   - local: HA local Assist pipeline if already configured

## Phase 5 — Configure Android phone
1. Install/update **Home Assistant Companion App** on the spare Android phone.
2. Log into the Home Assistant server.
3. In the app: **Settings → Companion app → Assist for Android**.
4. Tap **Set as default**.
5. In Android system settings, set **Home Assistant** as the default digital assistant app.
6. Confirm assistant launch works via:
   - swipe assistant gesture, or
   - long-press power/home depending on phone
7. Confirm it works from lock screen.

## Phase 6 — Enable Android wake word
1. In Home Assistant app: **Settings → Companion app → Assist for Android**.
2. Enable **Wake word detection**.
3. Choose one of the available wake words:
   - Hey Nabu
   - Hey Jarvis
   - Hey Mycroft
4. Say the wake word, wait for the listening prompt, then ask the command.

Notes:
- Android wake word is experimental.
- It works in background/locked mode per HA docs once enabled.
- It uses more battery than “Ok Google” because third-party apps do not get Google's low-power wake-word hardware path.
- Keep the spare phone plugged in.

## Proof-of-concept tests
Run these in order:
1. “Hey Jarvis, what time is it?”
2. “Hey Jarvis, turn on/off a safe test light/device.”
3. “Hey Jarvis, what is the state of [known HA entity]?”
4. “Hey Jarvis, remember that I’m testing the kitchen assistant.”
5. Follow-up: “What am I testing?”
6. “Hey Jarvis, summarize what you can control in this house.”
7. Lock the phone, wait 2 minutes, repeat a command.
8. Leave it plugged in for 1–2 hours and check wake reliability + battery/heat.

## Success criteria
- Wake word triggers reliably while locked/backgrounded.
- End-to-end answer latency feels acceptable.
- Hermes gets enough context to do useful actions.
- HA device control works through Hermes, not only HA's local command handler.
- Failure mode is recoverable without force-stopping the app.

## Likely failure points
- Hermes API not reachable from HA.
- Self-signed HTTPS/cert settings wrong.
- HA voice pipeline not configured before Android wake-word setup.
- Android battery optimization kills wake detection.
- “Prefer handling commands locally” bypasses Hermes.
- Hermes lacks a Home Assistant token/tool configuration, so it can chat but not act.

## Recommendation
Start with the HA add-on path if the goal is fastest working proof-of-concept. If that works but lacks access to Ayush's main Hermes environment/tools, then decide whether to migrate to standalone Hermes API or bridge the existing Hermes instance into HA.
