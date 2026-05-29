# Call Bullshit

A real-time live fact-check voice agent. It listens to a speaker, extracts
checkable claims on the fly, searches the web for evidence, and **interrupts
out loud** the moment it catches a lie.

**Built in a 5-hour hackathon in Berlin (29 May 2026), hosted by [Nebius](https://nebius.com/), [Tavily](https://tavily.com/), and [Gradium](https://gradium.ai/).** The stack uses all three hosts' APIs: Gradium for hearing and speaking, Tavily for live web evidence, Nebius for the LLM legs (claim extraction, verdict judging, rebuttal copy).

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
  🧠 Nebius LLM (Qwen3-30B MoE)             🔍 Tavily Search
     • extract claim from transcript           up to 5 results + synthesized answer
     • judge verdict (contradicted /           judge reads **all** fetched snippets
       supported / mixed / unknown)
     • generate spoken rebuttal (matches barker voice)
       │
       │  contradicted + confidence ≥ 0.5
       ▼
  📢 Voice Interrupt
     1. Barker (canned opener, plays immediately — zero gen latency)
     2. Rebuttal (Nebius gen + **streaming** Gradium TTS, buffered during barker)
       │  WebSocket events
       ▼
  📊 Live Dashboard              BS meter · live transcript · verdict feed · latency strip
```

Open `slide.html` in a browser for the visual architecture diagram.

---

## Features

| Feature | What it does |
|---------|----------------|
| **Mid-speech fact-checks** | Dispatches a new check every ~15 words (`LIVE_CHECK_WORDS`) so a verdict can land before the speaker finishes. |
| **Verdict-driven heckle** | Fires on `contradicted` with confidence ≥ 0.5 — not LLM whim. The BS meter is a **dashboard visual** (rises on dubious verdicts, resets after a heckle). |
| **Barker-as-latency-budget** | A pre-generated opener plays while Nebius + Gradium prepare the rebuttal. Barker length is sized to cover **gen + time-to-first TTS chunk**, not full synthesis. |
| **Warm startup calibration** | Two mock rebuttal rounds at launch: discard the cold one, size barkers from the warm **first-chunk** latency (+ margin). Concurrent `warm_up_verdict_path()` hides Nebius cold-start on the first real claim. |
| **Full Tavily evidence to judge** | Fetches up to 5 results (`TAVILY_MAX_RESULTS`); every snippet is fed to the judge (capped at `SNIPPET_CHARS` each). Latency is flat across 3–5 results. |
| **Fast MoE on all LLM legs** | Default `Qwen/Qwen3-30B-A3B-Instruct-2507` for extract, judge, and rebuttal (~3× faster than the original Llama-70B path). Env-overridable per leg. |
| **Streaming rebuttal TTS** | `tts_stream` + chunked playback — audio starts on the first PCM chunk (~600 ms warm) instead of waiting for a full WAV. |
| **Heuristic fallback** | If `judge_verdict` fails, `score_verdict` derives a verdict from Tavily's answer string without a second LLM call. |
| **Live dashboard** | WebSocket UI: BS meter, rolling transcript, verdict cards, red flash on heckle, per-stage latency breakdown. |

**Key design choices:**

- **Code-driven loop** — Python owns when to search, judge, and interrupt; the LLM does not decide to "call a tool."
- **Evidence-grounded** — Tavily supplies facts; the Nebius judge only classifies that evidence (with a heuristic fallback if the judge call fails).
- **Do not use reasoning / `*-Thinking*` models** for the three LLM legs — they exhaust `max_tokens` on internal traces and return empty TTS input.

---

## Partners (hackathon hosts)

| Service | Role | Links |
|---------|------|--------|
| **Gradium** | Streaming STT (semantic VAD) + streaming TTS (10 voices for barkers and rebuttals) | [gradium.ai](https://gradium.ai/) · [@gradium-ai](https://github.com/gradium-ai) |
| **Nebius** | LLM inference — claim extraction, verdict judging, rebuttal generation (default `Qwen/Qwen3-30B-A3B-Instruct-2507`) | [nebius.com](https://nebius.com/) · [@nebius](https://github.com/nebius) |
| **Tavily** | Real-time web search — evidence and synthesized answer for the judge | [tavily.com](https://tavily.com/) · [@tavily-ai](https://github.com/tavily-ai) |

Python SDKs in this repo: [`gradium`](https://pypi.org/project/gradium/), [`tavily-python`](https://github.com/tavily-ai/tavily-python), [`openai`](https://github.com/openai/openai-python) (Nebius-compatible API).

---

## Files

```
main.py            entry point — mic loop, STT, parallel checks, barker/rebuttal pipeline, dashboard WS
factcheck.py       fact-checking brain — claim extraction, Tavily, judge, rebuttal, TTS adapters
gen_barkers.py     one-shot script to generate the barker WAV library
dashboard.html     live browser dashboard (WebSocket)
slide.html         architecture slide for demos
test_factcheck.py  unit tests for pure functions (no network)
test_main.py       unit tests for calibration, streaming rebuttal prep, barker pick
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

Optional tuning (see `.env.example` for defaults):

```
# Models (all three legs; use *-Instruct*, not *-Thinking*)
JUDGE_MODEL=...
EXTRACT_MODEL=...
REBUTTAL_MODEL=...

# Tavily → judge
TAVILY_MAX_RESULTS=5
SNIPPET_CHARS=400

# STT / live checks
VAD_THRESHOLD=0.7
VAD_STEPS_TO_END=8
LIVE_CHECK_WORDS=15
LIVE_CHECK_MIN_WORDS=10

# BS meter (visual only; interrupt is verdict-driven)
BS_THRESHOLD=1.0
BS_DECAY=0.3

# Barker sizing
CALIBRATION_MARGIN=1.5
CALIBRATION_MAX=17.0
CALIBRATE=1              # 0 = skip startup calibration

# Streaming TTS playback
TTS_PREBUFFER_CHUNKS=2
MAX_REBUTTAL_SECONDS=30

DASHBOARD_PORT=8765
INPUT_DEVICE=            # substring match (e.g. "JBL") or device index
DEBUG=0
```

---

## Run

```bash
# Generate barker audio (one-time)
python gen_barkers.py

# Start the agent (serves dashboard on DASHBOARD_PORT)
python main.py
```

Open `http://localhost:8765` in a browser to watch the live dashboard.

Run tests:

```bash
pytest test_factcheck.py test_main.py -v
```

---

## License

[MIT](LICENSE) — use, fork, and adapt freely; attribution appreciated.

## Contributing

Built in a weekend hackathon window — not a maintained product, but fixes and ideas are welcome:

- [Open an issue](https://github.com/timotif/call-bullshit/issues) for bugs or questions
- [Open a pull request](https://github.com/timotif/call-bullshit/pulls) for small, focused changes

No formal contribution guide; keep PRs scoped and runnable (`pytest` passes).

---

If this demo was useful or made you smile, a [⭐ on GitHub](https://github.com/timotif/call-bullshit) helps others find it.
