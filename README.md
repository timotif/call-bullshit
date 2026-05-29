# Call Bullshit

A real-time live fact-check voice agent. It listens to a speaker, extracts
checkable claims on the fly, searches the web for evidence, and **interrupts
out loud** the moment it catches a lie.

---

## Architecture

```
  🎙️ Microphone
       │  24 kHz PCM / 80 ms frames
       ▼
  👂 Gradium STT  ──────────────── streaming transcript + VAD turn detection
       │  live text (every segment revision)
       ▼
  ⚡ Parallel Fact-Check          dispatch every ~15 words while still speaking
       │  extracted claim
       ├──────────────────────────────────────────┐
       ▼                                          ▼
  🧠 Nebius LLM Judge                        🔍 Tavily Search
     • extract claim from transcript           real-time web evidence
     • judge verdict (contradicted /           returns answer + top snippets
       supported / mixed / unknown)
     • generate spoken rebuttal
       │
       │  verdict: contradicted (conf ≥ 0.5)
       ▼
  📢 Voice Interrupt
     1. Barker (canned opener, plays immediately — zero gen latency)
     2. Rebuttal (Nebius gen + Gradium TTS, runs concurrently with barker)
       │  WebSocket events
       ▼
  📊 Live Dashboard              BS meter · live transcript · verdict feed
```

**Key design choices:**

- **Parallel mid-speech checks** — fact-check tasks are dispatched every ~15
  words so a verdict can arrive *before* the turn ends.
- **Barker-as-latency-budget** — the ~4–17 s canned opener plays while Nebius
  and Gradium TTS prepare the rebuttal concurrently, hiding network latency
  completely.
- **Startup calibration** — one mock rebuttal round measures the session's
  actual gen+TTS latency and picks the shortest barker that still covers it.
- **Heuristic fallback** — if the LLM judge call fails, `score_verdict` derives
  a verdict from Tavily's answer string without a second network call.

Open `slide.html` in a browser for the visual architecture diagram.

---

## Partners

| Service | Role |
|---------|------|
| **Gradium** | Streaming STT (VAD turn detection) + TTS (10 voices) |
| **Nebius** | LLM inference — claim extraction, verdict judging, rebuttal generation (`meta-llama/Llama-3.3-70B-Instruct`) |
| **Tavily** | Real-time web search — the sole source of truth for verdicts |

---

## Files

```
main.py            entry point — mic loop, STT, BS meter, barker/rebuttal pipeline
factcheck.py       fact-checking brain — claim extraction, verdict, rebuttal, TTS
gen_barkers.py     one-shot script to generate the barker WAV library
dashboard.html     live browser dashboard (WebSocket)
slide.html         architecture slide
test_factcheck.py  unit tests for all pure functions (no network)
```

> `barkers/` is generated locally by `gen_barkers.py` and not checked in.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

Required env vars:
```
GRADIUM_API_KEY=...
NEBIUS_API_KEY=...
TAVILY_API_KEY=...
```

Optional tuning:
```
VAD_THRESHOLD=0.7        # inactivity probability to end a turn
VAD_STEPS_TO_END=8       # consecutive high-VAD frames before flush (~640 ms)
LIVE_CHECK_WORDS=15      # dispatch a new parallel check every N new words
LIVE_CHECK_MIN_WORDS=10  # minimum words before first live check
BS_THRESHOLD=1.0         # BS meter threshold (cosmetic; interrupt is verdict-driven)
CALIBRATION_MARGIN=1.5   # headroom (s) added to measured gen+tts latency
CALIBRATION_MAX=17.0     # cap on barker budget (s)
CALIBRATE=1              # set to 0 to skip startup latency calibration
DASHBOARD_PORT=8765
INPUT_DEVICE=            # substring match (e.g. "JBL") or device index; unset = system default
DEBUG=                   # set to 1 for verbose STT/chunk logging
```

---

## Run

```bash
# Generate barker audio (one-time)
python gen_barkers.py

# Start the agent
python main.py
```

Open `http://localhost:8765` in a browser to watch the live dashboard.

Run tests:
```bash
pytest test_factcheck.py -v
```
