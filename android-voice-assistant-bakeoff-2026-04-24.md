# Android Alexa-like Assistant Bake-off — 2026-04-24

## Goal
Turn a spare Android phone into an Alexa-like intelligent assistant:
- wake word or equivalent assistant gesture
- ask naturally
- assistant actually does things
- voice reply
- ideally self-hostable or at least not locked to one cloud assistant

## Contenders
1. OpenClaw Assistant + OpenClaw
2. Home Assistant Assist on Android
3. Sanna
4. Built-in Google Assistant (baseline)
5. Home Assistant Assist + Hermes HA integration (`WolframRavenwolf/hermes-ha-integration`)

## Evidence highlights

### OpenClaw Assistant + OpenClaw
- Important correction: the **latest official OpenClaw Android docs do not confirm Android wake word availability** in the official/core Android app.
- Official docs currently say Android uses a single mic on/off Voice-tab flow, local TTS fallback, and voice stops when the app leaves the foreground.
- Official docs also explicitly say: “Voice wake/talk-mode toggles are currently removed from Android UX/runtime.”
- Official docs still state the main Android app has not been publicly released in the core repo and can be built from source there.
- The separate `yuga-hashimoto/openclaw-assistant` README/release does claim offline wake word detection via Vosk, customizable wake words, continuous conversation mode, system assistant integration, auto-start on boot, and battery optimization exclusion.
- Therefore OpenClaw wake word is **not confirmed from official OpenClaw Android docs**; it is only claimed by the separate OpenClaw Assistant app. This must be validated on-device before treating OpenClaw as the top option.

### Home Assistant Assist on Android
- Official HA docs confirm it can be set as the default digital assistant app.
- Can be launched from assistant gesture/home/power and from lock screen.
- Wake word detection is available, processed locally on-device with microWakeWord.
- Wake word remains experimental and has noticeable battery impact.
- Best fit when the actions you want are represented inside Home Assistant.

### Sanna
- README positions it as an open-source voice-first Android assistant that actually controls the phone.
- Claims wake word -> STT -> LLM -> TTS, Android Accessibility automation, app control, scheduler/sub-agents, lists/journal, and no backend.
- Sanity-check result: likely **not** the right primary path for this project.
- Distribution is weak today: beta APK via email or build from source; README says it is best suited for technical beta testers.
- Setup burden is high: OpenAI/Claude keys, Spotify client ID, Google OAuth/web client + Gmail/Calendar/Tasks/People APIs + SHA-1 registration, Picovoice access key, Slack OAuth redirect setup, Google Maps API key, Brave Search API key, etc.
- Its architecture is a phone-local personal automation agent, not a Hermes-backed always-on appliance. It may be powerful, but it would make the spare phone itself the core agent runtime and credential store.
- Repo is much earlier/smaller than the other serious options: 12 stars, 2 forks, 1 issue at sanity-check time, beta/tester-stage.
- Keep only as an experimental side branch if phone-local Accessibility automation becomes the main requirement.

### Built-in Google Assistant
- Official Google docs confirm “Hey Google” voice activation with Voice Match and lock-screen support on supported Android devices.
- Easiest zero-build baseline.
- Strong assistant UX, but limited if the real requirement is deep custom action execution and model/provider flexibility.

### Home Assistant Assist + Hermes HA integration
- `WolframRavenwolf/hermes-ha-integration` is a Home Assistant custom integration that makes Hermes Agent a Home Assistant **conversation agent** for HA voice assistants and the conversation panel.
- It connects to a running Hermes Agent API server via OpenAI-compatible `/v1/chat/completions`.
- Claimed features: streaming, entity exposure in the system prompt, multi-turn conversation history, username resolution, multiple Hermes instances.
- Companion repo `WolframRavenwolf/hermes-ha-addon` packages Hermes Agent as a Home Assistant add-on with an OpenAI-compatible API, dashboard, terminal, and Home Assistant token configuration.
- This does not replace the Android wake-word layer; it improves the **Home Assistant Assist** route by letting Android Assist use Hermes as the brain.
- Main risk: integration/add-on maturity is smaller/newer than HA itself, and it depends on Hermes API reliability inside HA.

## Weighted score
Criteria weights:
- Alexa-like UX: 25%
- Real actions: 25%
- Model flexibility: 15%
- Setup friction: 15%
- Self-host/privacy: 10%
- Maturity: 10%

Scores (/10):
- OpenClaw Assistant: 8.10
- Home Assistant Assist + Hermes HA integration: 7.75
- Home Assistant Assist: 7.35
- Sanna: 7.35
- Built-in Google Assistant: 7.10

## Decision
### Test first
**OpenClaw Assistant + OpenClaw**
Reason: best balance of wake word, voice loop, actual phone/device actions, self-hosting, and provider flexibility.

### Test second
**Home Assistant Assist + Hermes HA integration**
Reason: this may be the cleanest way to combine Android's HA wake-word/default-assistant support with Hermes as the actual conversation brain and action executor. It is especially attractive if the spare phone is meant to control home/server/Hermes workflows more than arbitrary phone-local apps.

### Test third
**Sanna**
Reason: very promising for direct phone automation, but too early to make the primary path yet.

### Keep as baseline only
**Built-in Google Assistant**
Reason: easiest appliance behavior, but not the best path if we want a truly programmable intelligent assistant.

## Recommended proof-of-concept checks
Each candidate should be tested on the same tasks:
1. Wake from across the room.
2. Work while locked / screen off if possible.
3. Answer a factual question.
4. Perform an actual action: send a message, create a reminder, or control a device.
5. Handle a follow-up question in the same conversation.
6. Recover cleanly from a bad transcription.
7. Measure rough latency and battery impact.

## Final recommendation
Do not build custom voice infrastructure first.
Run a practical proof-of-concept with OpenClaw first. If that is too immature or awkward, the next-best path is **Home Assistant Assist + Hermes HA integration**, because it uses HA's Android wake-word/default-assistant path while delegating conversation/action intelligence to Hermes. Only then decide whether custom Hermes/PWA work is justified.
