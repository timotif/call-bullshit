# Call Bullshit

**Real-time voice fact-checker — listens while you speak, searches the web, and talks over you when you're wrong.**

Warm path: **~2 s** from a `contradicted` verdict to hearing the correction — a short ~1–2 s opener plus streaming TTS that starts on the first PCM chunk (not a long canned ramble waiting for a full WAV).

Built in a **5-hour hackathon** in Berlin (29 May 2026) with [Nebius](https://nebius.com/), [Tavily](https://tavily.com/), and [Gradium](https://gradium.ai/). Gradium for hearing and speaking, Tavily for live web evidence, Nebius for the LLM legs (claim extraction, verdict judging, rebuttal copy).

---

## Demo

<video src="https://github.com/user-attachments/assets/fa03d405-d31f-4d11-b369-d7fdf1864ff8" width="720" controls />

`contradicted` on the [live dashboard](http://localhost:8765) → short barker → rebuttal

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # GRADIUM_API_KEY, NEBIUS_API_KEY, TAVILY_API_KEY

python gen_barkers.py         # one-time: generate barker WAVs (includes ~1–2 s tier)
python main.py                # listen — dashboard on http://localhost:8765
```

First run runs a one-time warm-up calibration (~5 s); then speak. Use `*-Instruct*` models only — not `*-Thinking*` (see [Configuration](#configuration)).

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

Open [`slide.html`](slide.html) for a visual architecture slide.

---

## Features

| Feature | What it does |
|---------|----------------|
| **Mid-speech fact-checks** | New check every ~15 words (`LIVE_CHECK_WORDS`) so a verdict can land before the speaker finishes. |
| **Verdict-driven heckle** | Fires on `contradicted` with confidence ≥ 0.5 — not LLM whim. The BS meter is a **dashboard visual** (rises on dubious verdicts, resets after a heckle). |
| **Barker-as-latency-budget** | ~1–2 s openers cover **gen + time-to-first TTS chunk** (~1.2 s warm); longer tiers only when calibration sees a slow round. |
| **Warm startup calibration** | Two mock rounds at launch: discard the cold one, size barkers from warm **first-chunk** latency (+ margin). `warm_up_verdict_path()` hides Nebius cold-start on the first real claim. |
| **Full Tavily evidence to judge** | Up to 5 results (`TAVILY_MAX_RESULTS`); every snippet to the judge (`SNIPPET_CHARS` each). Latency is flat across 3–5 results. |
| **Fast MoE on all LLM legs** | Default `Qwen/Qwen3-30B-A3B-Instruct-2507` (~3× faster than the original Llama-70B path). Env-overridable per leg. |
| **Streaming rebuttal TTS** | `tts_stream` + chunked playback — first PCM chunk ~600 ms warm, not a full WAV. |
| **Heuristic fallback** | If `judge_verdict` fails, `score_verdict` uses Tavily's answer string — no second LLM call. |
| **Live dashboard** | BS meter, rolling transcript, verdict cards, heckle flash, per-stage latency. |

**Key design choices:**

- **Code-driven loop** — Python owns when to search, judge, and interrupt; the LLM does not decide to "call a tool."
- **Evidence-grounded** — Tavily supplies facts; the Nebius judge only classifies that evidence.
- **Do not use `*-Thinking*` / reasoning models** on the three LLM legs — they exhaust `max_tokens` on internal traces and return empty TTS input.

---

## Configuration

Required in `.env`:

```
GRADIUM_API_KEY=...
NEBIUS_API_KEY=...
TAVILY_API_KEY=...
```

Common tuning (full list in [`.env.example`](.env.example)):

| Area | Variables |
|------|-----------|
| Models | `JUDGE_MODEL`, `EXTRACT_MODEL`, `REBUTTAL_MODEL` |
| Tavily → judge | `TAVILY_MAX_RESULTS`, `SNIPPET_CHARS` |
| Live checks | `LIVE_CHECK_WORDS`, `LIVE_CHECK_MIN_WORDS`, `VAD_*` |
| BS meter (visual) | `BS_THRESHOLD`, `BS_DECAY` |
| Barker sizing | `CALIBRATE`, `CALIBRATION_MARGIN`, `CALIBRATION_MAX` |
| Streaming playback | `TTS_PREBUFFER_CHUNKS`, `MAX_REBUTTAL_SECONDS` |
| Other | `DASHBOARD_PORT`, `INPUT_DEVICE`, `DEBUG` |

---

## Partners (hackathon hosts)

| Service | Role | Links |
|---------|------|--------|
| **Gradium** | Streaming STT (semantic VAD) + streaming TTS (10 voices) | [gradium.ai](https://gradium.ai/) · [@gradium-ai](https://github.com/gradium-ai) |
| **Nebius** | Claim extraction, verdict judging, rebuttal (`Qwen/Qwen3-30B-A3B-Instruct-2507`) | [nebius.com](https://nebius.com/) · [@nebius](https://github.com/nebius) |
| **Tavily** | Real-time search + synthesized answer for the judge | [tavily.com](https://tavily.com/) · [@tavily-ai](https://github.com/tavily-ai) |

SDKs: [`gradium`](https://pypi.org/project/gradium/), [`tavily-python`](https://github.com/tavily-ai/tavily-python), [`openai`](https://github.com/openai/openai-python) (Nebius-compatible).

---

## Project layout

```
main.py            mic loop, STT, parallel checks, interrupt pipeline, dashboard WS
factcheck.py       claim extraction, Tavily, judge, rebuttal, TTS adapters
gen_barkers.py     generate barker WAV library (one-time)
dashboard.html     live dashboard
slide.html         architecture slide
test_factcheck.py  unit tests (no network)
test_main.py       calibration, streaming rebuttal prep, barker pick
assets/demo.mp4    optional — drop your demo video here
```

> `barkers/` is generated by `gen_barkers.py` and not checked in.

---

## Tests

```bash
pytest test_factcheck.py test_main.py -v
```

---

## License

[MIT](LICENSE) — use, fork, and adapt freely; attribution appreciated.

## Contributing

Built in a **5-hour hackathon** — not a maintained product, but fixes and ideas are welcome ([issues](https://github.com/timotif/call-bullshit/issues) · [PRs](https://github.com/timotif/call-bullshit/pulls)). Keep PRs scoped; `pytest` should pass.

---

If this demo was useful or made you smile, a [⭐ on GitHub](https://github.com/timotif/call-bullshit) helps others find it.
