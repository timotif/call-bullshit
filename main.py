"""Call Bullshit — live voice fact-check agent.

Run: .venv/bin/python main.py
Speak into the mic. Each finalized turn is fact-checked (Nebius claim extraction
+ Tavily verdict). A contradicted claim fires a barker interruption followed by
a Nebius-generated spoken rebuttal.
"""

import asyncio
import json
import os
import queue
import random
import sys
import time
import wave
from pathlib import Path

import aiohttp
import aiohttp.web
import gradium
import sounddevice as sd
from dotenv import load_dotenv

from factcheck import check_turn, generate_rebuttal, play_audio, speak, warm_up_verdict_path

load_dotenv()

_REQUIRED_ENV = ("GRADIUM_API_KEY", "NEBIUS_API_KEY", "TAVILY_API_KEY")
_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    sys.exit(f"Missing required environment variables: {', '.join(_missing)}\nCopy .env.example to .env and fill in your keys.")

WAV_HEADER_BYTES = 44  # canonical RIFF/WAV header size; Gradium WAVs use standard 44-byte headers
SAMPLE_RATE = 24000  # Gradium "pcm": 24 kHz, 16-bit signed LE, mono
FRAME_SAMPLES = 1920  # 80 ms
# Noisy-room tuning (see Gradium turn-taking guide: "require several consecutive
# high-confidence steps"). Raise THRESHOLD so only confident silence counts; raise
# STEPS_TO_END so a brief noise dip doesn't end a turn prematurely.
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.7"))
VAD_STEPS_TO_END = int(os.environ.get("VAD_STEPS_TO_END", "8"))  # ~640 ms silence
# Live parallel fact-check: dispatch a new check every N new words while speaking.
# Lower = more checks in parallel but more API calls; higher = less responsive.
LIVE_CHECK_WORDS = int(os.environ.get("LIVE_CHECK_WORDS", "15"))
LIVE_CHECK_MIN_WORDS = int(os.environ.get("LIVE_CHECK_MIN_WORDS", "10"))

# BS meter: rises on dubious verdicts, decays on clean turns. Crossing the
# threshold fires a barker. Tuned for demo responsiveness, override via env.
BS_THRESHOLD = float(os.environ.get("BS_THRESHOLD", "1.0"))
BS_DECAY = float(os.environ.get("BS_DECAY", "0.3"))  # subtracted per clean turn
BARKERS_DIR = Path(__file__).parent / "barkers"
DEBUG = bool(os.environ.get("DEBUG"))

# Headroom (s) added to the calibrated prep latency when sizing the barker, so
# normal per-call variance doesn't leak dead-air. Set CALIBRATE=0 to skip the
# startup mock round (uses the longest barker, safest but most talk-time).
CALIBRATION_MARGIN = float(os.environ.get("CALIBRATION_MARGIN", "1.5"))
CALIBRATION_MAX = float(os.environ.get("CALIBRATION_MAX", "17.0"))  # never pick a barker longer than this
CALIBRATE = os.environ.get("CALIBRATE", "1") != "0"
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))
DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"

# ── Dashboard websocket broadcaster ─────────────────────────────────────────
_ws_clients: set[aiohttp.web.WebSocketResponse] = set()
# Last status event — replayed to any browser that connects after it was emitted.
_last_status: dict | None = None


def emit(event_type: str, payload: dict) -> None:
    """Fire-and-forget: push a JSON event to all connected dashboard browsers.

    Safe to call from any async coroutine on the main loop — creates a task
    so it never blocks the caller. Silently drops failed/closed clients.
    """
    global _last_status
    if event_type == "status":
        _last_status = payload  # remember for late-connecting browsers
    if not _ws_clients:
        return
    try:
        msg = json.dumps({"type": event_type, **payload})
    except (TypeError, ValueError) as exc:
        log(f"[ws] emit serialization error ({event_type}): {exc}")
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_broadcast(msg))
    except RuntimeError:
        pass  # no running loop (e.g., called during shutdown)


async def _broadcast(msg: str) -> None:
    for ws in list(_ws_clients):
        try:
            if not ws.closed:
                await ws.send_str(msg)
        except Exception as exc:
            log(f"[ws] broadcast error: {exc}")
            _ws_clients.discard(ws)


async def _ws_handler(request: aiohttp.web.Request) -> aiohttp.web.WebSocketResponse:
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    _ws_clients.add(ws)
    # Replay the last status so a browser opened mid-calibration sees it immediately.
    if _last_status is not None:
        try:
            await ws.send_str(json.dumps({"type": "status", **_last_status}))
        except Exception:
            pass
    try:
        async for _ in ws:
            pass  # we only push; ignore any client messages
    finally:
        _ws_clients.discard(ws)
    return ws


async def _html_handler(_request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.Response(
        text=DASHBOARD_HTML.read_text(),
        content_type="text/html",
    )


async def start_dashboard() -> None:
    app = aiohttp.web.Application()
    app.router.add_get("/", _html_handler)
    app.router.add_get("/ws", _ws_handler)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "0.0.0.0", DASHBOARD_PORT)
    await site.start()
    print(f"[dashboard] http://localhost:{DASHBOARD_PORT}  (ws on /ws)", flush=True)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


# Order in which stages are reported; only present keys are shown.
_STAGE_ORDER = [
    ("extract_claim", "claim"),
    ("tavily", "search"),
    ("judge", "judge"),
    ("rebuttal_gen", "rebut-gen"),
    ("tts", "tts"),
    ("barker_play", "barker"),
    ("rebuttal_play", "rebut-play"),
]


def log_timings(timings: dict) -> None:
    """Print a per-stage latency breakdown for one turn, blaming the slow stage.

    Headline totals for placing blame:
      - verdict_total: turn-end -> verdict ready (the 'caught you' delay):
          claim + search + judge
      - rebuttal_gap:  barker fires -> rebuttal audio ready (dead air after the
          heckle); ideally <= barker_play so it's fully hidden.
    """
    parts = [f"{label}={timings[key]*1000:.0f}ms" for key, label in _STAGE_ORDER if key in timings]
    verdict_total = sum(timings.get(k, 0.0) for k in ("extract_claim", "tavily", "judge"))
    timings["verdict_total"] = verdict_total  # write back so emit payload includes it
    totals = [f"verdict_total={verdict_total*1000:.0f}ms"]
    if "rebuttal_gap" in timings:
        gap = timings["rebuttal_gap"]
        bark = timings.get("barker_play", 0.0)
        hidden = "hidden" if gap <= bark else f"+{(gap-bark)*1000:.0f}ms dead-air"
        totals.append(f"rebuttal_gap={gap*1000:.0f}ms ({hidden})")
    if "bark_rebuttal_overlap" in timings:
        ov = timings["bark_rebuttal_overlap"]
        rebut_work = timings.get("rebuttal_gen", 0.0) + timings.get("tts", 0.0)
        bark = timings.get("barker_play", 0.0)
        # Truly parallel => overlap ~= min(bark, rebut_work). Serial => ~0.
        expected = min(bark, rebut_work)
        verdict = "PARALLEL" if ov > 0.5 * expected and expected > 0 else "SERIAL"
        totals.append(f"overlap={ov*1000:.0f}ms/{expected*1000:.0f}ms [{verdict}]")
    # Slowest network stage = the one to attack first.
    net = {k: v for k, v in timings.items() if k in {"extract_claim", "tavily", "judge", "rebuttal_gen", "tts"}}
    blame = max(net, key=net.get) if net else None
    timings["slowest"] = blame  # write back for dashboard
    blame_str = f"  ⟵ slowest: {blame}={net[blame]*1000:.0f}ms" if blame else ""
    log("[latency] " + "  ".join(parts + totals) + blame_str)


async def prepare_rebuttal(verdict: dict, timings: dict | None = None, voice_id: str | None = None, barker_text: str | None = None) -> tuple[str, bytes]:
    """Generate rebuttal text (Nebius) and synthesize speech (Gradium TTS).

    No playback — returns (text, wav_bytes) so playback can be sequenced after
    the barker. Runs concurrently with barker playback to hide latency.
    If `timings` is passed, records `rebuttal_gen` and `tts` durations (s).
    `voice_id` lets the rebuttal match the barker's voice.
    `barker_text` is passed to the LLM so the rebuttal continues the opener seamlessly.
    """
    loop = asyncio.get_running_loop()
    if timings is not None:
        timings["rebuttal_start"] = time.perf_counter()  # absolute: when prep work began
    t0 = time.perf_counter()
    rebuttal = await loop.run_in_executor(None, generate_rebuttal, verdict, barker_text)
    if timings is not None:
        timings["rebuttal_gen"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    wav = await speak(rebuttal, voice_id=voice_id) if voice_id else await speak(rebuttal)
    if timings is not None:
        timings["tts"] = time.perf_counter() - t1
        timings["rebuttal_end"] = time.perf_counter()  # absolute: prep work done
    return rebuttal, wav


async def calibrate_prep_latency() -> float:
    """Mock two rebuttals (gen + TTS) at startup to size barkers for this session.

    The FIRST round is a throwaway warm-up: the rebuttal model (70B) and Gradium
    TTS both pay a cold-start on their first request after idle, so measuring a
    cold round would over-estimate the budget and pick barkers longer than real
    fires need. We discard the first round and calibrate on the SECOND (warm) one,
    so the budget reflects steady-state and CALIBRATION_MARGIN only covers genuine
    per-call variance.

    Returns the prep latency budget in seconds: measured warm (gen + tts) + margin.
    Returns 0.0 if calibration is disabled or fails — play_barker then falls back
    to the shortest barker (budget=0) or, on a real fire, the longest cover.
    """
    if not CALIBRATE:
        return 0.0
    mock_verdict = {
        "claim": "The Great Wall of China is visible from space with the naked eye.",
        "status": "contradicted",
        "confidence": 0.9,
        "summary": "It is not visible from space with the naked eye; this is a common myth.",
    }
    print("Calibrating rebuttal latency (warm-up + measured round)...", flush=True)
    emit("status", {"phase": "calibrating", "text": "Calibrating latency…"})
    timings: dict[str, float] = {}
    try:
        await prepare_rebuttal(mock_verdict)        # cold throwaway: warms gen + TTS
        await prepare_rebuttal(mock_verdict, timings)  # warm: this is what we measure
    except Exception as exc:
        print(f"[WARN] calibration failed ({exc}); using longest barker", flush=True)
        emit("status", {"phase": "calibration_failed", "text": f"Calibration failed: {exc}"})
        return 0.0
    prep = timings.get("rebuttal_gen", 0.0) + timings.get("tts", 0.0)
    budget = min(prep + CALIBRATION_MARGIN, CALIBRATION_MAX)
    capped = " (capped)" if prep + CALIBRATION_MARGIN > CALIBRATION_MAX else ""
    print(
        f"Calibrated: gen+tts={prep:.1f}s, barker budget={budget:.1f}s "
        f"(+{CALIBRATION_MARGIN:.1f}s margin{capped})\n",
        flush=True,
    )
    emit("status", {
        "phase": "ready",
        "text": f"Ready — gen+tts={prep:.1f}s, barker budget={budget:.1f}s{capped}",
        "prep_s": prep,
        "budget_s": budget,
    })
    return budget


def verdict_delta(verdict: dict) -> float:
    """How much a verdict moves the BS meter. contradicted >> mixed; clean/unknown <= 0."""
    status = verdict.get("status")
    conf = verdict.get("confidence", 0.5)
    if status == "contradicted":
        return conf  # up to ~0.8
    if status == "mixed":
        return 0.4 * conf  # hedged false claims still nudge it
    if status == "supported":
        return -BS_DECAY
    return 0.0  # unknown: no signal


def _barker_duration(path: Path) -> float:
    """Playback duration (s) from the WAV header + true PCM payload size.

    Gradium WAVs carry an unreliable RIFF size, so compute from the file's actual
    byte length minus the 44-byte canonical header — matching how play_barker reads it.
    """
    with wave.open(str(path), "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
    pcm_bytes = path.stat().st_size - WAV_HEADER_BYTES
    return pcm_bytes / (sr * ch * sw)


def load_barkers() -> list[dict]:
    """Load the barker manifest, tagging each entry with its playback duration (s)."""
    manifest = BARKERS_DIR / "manifest.json"
    if not manifest.exists():
        return []
    barkers = json.loads(manifest.read_text())
    for b in barkers:
        file = Path(b["file"])
        if file.parent != Path(".") or file.suffix.lower() != ".wav":
            raise ValueError(f"Unsafe barker path in manifest: {b['file']!r}")
        b["duration"] = _barker_duration(BARKERS_DIR / file)
    return barkers


# Shuffle queues keyed by frozenset of candidate file names, so each eligible
# pool gets its own independent non-repeating sequence.
_barker_queues: dict[frozenset, list[dict]] = {}
_barker_last: dict[frozenset, str] = {}  # last file played per pool key


def pick_barker(barkers: list[dict], budget: float) -> dict:
    """Pick a random barker whose duration covers `budget` (the prep latency).

    Uses a shuffle-queue per eligible pool so the same barker is never picked
    twice until all others in the pool have been played. On pool reset, ensures
    the first pick of the new cycle doesn't repeat the last pick of the old one.
    If no barker covers the budget, falls back to the pool of longest-duration
    barkers.
    """
    covering = [b for b in barkers if b.get("duration", 0.0) >= budget]
    if covering:
        pool = covering
    else:
        max_dur = max(b.get("duration", 0.0) for b in barkers)
        pool = [b for b in barkers if b.get("duration", 0.0) == max_dur]

    key = frozenset(b["file"] for b in pool)
    if key not in _barker_queues or not _barker_queues[key]:
        shuffled = list(pool)
        random.shuffle(shuffled)
        # Avoid starting with the same barker that ended the previous cycle.
        last = _barker_last.get(key)
        if last and len(shuffled) > 1 and shuffled[-1]["file"] == last:
            shuffled[-1], shuffled[-2] = shuffled[-2], shuffled[-1]
        _barker_queues[key] = shuffled

    pick = _barker_queues[key].pop()
    _barker_last[key] = pick["file"]
    return pick


def play_barker(barkers: list[dict], budget: float = 0.0, chosen: dict | None = None) -> dict | None:
    """Play a barker WAV synchronously (blocking; caller runs it in a thread).

    If `chosen` is provided it is played directly (caller already called
    pick_barker for voice-matching). Otherwise picks via pick_barker(budget).
    Returns the played barker dict.
    """
    if not barkers:
        print("[barker] (no barker audio available)")
        return None
    pick = chosen if chosen is not None else pick_barker(barkers, budget)
    path = BARKERS_DIR / pick["file"]
    voice_label = pick.get("voice", "")
    print(f"\n[BARKER ~{pick.get('duration', 0):.0f}s voice={voice_label}] {pick['text']!r}")
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
    # Gradium WAVs carry an unreliable RIFF size, so wf.getnframes() under-reports.
    # Read the PCM payload directly: skip the canonical header.
    data = path.read_bytes()[WAV_HEADER_BYTES:]
    with sd.RawOutputStream(samplerate=sr, channels=ch, dtype="int16") as audio:
        audio.write(data)
    return pick


def mic_stream():
    """Yield 80 ms int16 mono PCM frames from the default input device."""
    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, _frames, _time, status):
        if status:
            print(f"[audio] {status}")
        q.put(bytes(indata))

    # INPUT_DEVICE: substring match (e.g. "iPhone", "MacBook", "JBL") or index.
    # Unset -> system default input.
    device = os.environ.get("INPUT_DEVICE") or None
    if device and device.isdigit():
        device = int(device)
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=1,
        device=device,
        callback=callback,
    )
    if device is not None:
        print(f"[mic] {sd.query_devices(stream.device)['name']}")
    return stream, q


async def main():
    api_key = os.environ["GRADIUM_API_KEY"]
    client = gradium.client.GradiumClient(api_key=api_key)

    barkers = load_barkers()
    bs_meter = 0.0
    # Guard: only one barker/rebuttal plays at a time. A turn that arrives while
    # speaking=True is still fact-checked and logged, but its audio is dropped.
    speaking = False

    loop = asyncio.get_running_loop()
    await start_dashboard()

    # Startup calibration: run one mock rebuttal (Nebius gen + Gradium TTS) to
    # measure THIS session's prep latency, so play_barker can pick the shortest
    # barker that still covers it. Network/API speed varies per session; measuring
    # beats hardcoding. CALIBRATION_MARGIN adds headroom for variance.
    # Warm the verdict-path models (extract + judge) concurrently so the first
    # REAL claim doesn't pay their cold-start; both legs hit Nebius, so overlap them.
    warm_task = loop.run_in_executor(None, warm_up_verdict_path)
    prep_budget = await calibrate_prep_latency()
    await warm_task

    stream, q = mic_stream()
    print(f"Listening @ {SAMPLE_RATE} Hz. Speak — Ctrl+C to stop.\n")

    async def fact_check_turn(text: str) -> None:
        """Off-thread fact-check + meter update; fires a barker on threshold."""
        nonlocal bs_meter, speaking
        loop = asyncio.get_running_loop()
        timings: dict[str, float] = {}
        verdict = await loop.run_in_executor(None, check_turn, text, timings)
        if verdict is None:
            bs_meter = max(0.0, bs_meter - BS_DECAY)  # no claim -> cool off
            print(f"   (no claim)  BS={bs_meter:.2f}")
            emit("meter", {"value": bs_meter})
            emit("no_claim", {"turn": text})
            log_timings(timings)
            return
        bs_meter = max(0.0, bs_meter + verdict_delta(verdict))
        fire = verdict["status"] == "contradicted" and verdict.get("confidence", 0) >= 0.5
        log(
            f"[{verdict['status']}] conf={verdict.get('confidence', 0):.2f}"
            f"  {verdict.get('source_title') or ''}  BS={bs_meter:.2f}"
        )
        log(f"   turn:  {text!r}")
        log(f"   claim: {verdict.get('claim')!r}")
        log(f"   facts: {verdict.get('summary', '')[:200]!r}")
        emit("meter", {"value": bs_meter})
        emit("verdict", {
            "status": verdict.get("status"),
            "confidence": verdict.get("confidence"),
            "source_title": verdict.get("source_title") or "",
            "claim": verdict.get("claim") or "",
        })
        if fire and speaking:
            log("[drop] already speaking — skipping barker for this turn")
            fire = False
        if fire:
            # Pre-pick barker so rebuttal can match its voice; then start both
            # concurrently — barker playback is the latency budget for rebuttal prep.
            chosen_barker = pick_barker(barkers, prep_budget) if barkers else None
            rebuttal_voice_id = chosen_barker.get("voice_id") if chosen_barker else None
            rebuttal_barker_text = chosen_barker.get("text") if chosen_barker else None
            speaking = True
            try:
                fire_t = time.perf_counter()
                rebuttal_task = asyncio.create_task(
                    prepare_rebuttal(verdict, timings, voice_id=rebuttal_voice_id, barker_text=rebuttal_barker_text)
                )
                bark_t = time.perf_counter()
                await loop.run_in_executor(None, play_barker, barkers, prep_budget, chosen_barker)
                bark_end = time.perf_counter()
                timings["barker_play"] = bark_end - bark_t
                rebuttal, wav = await rebuttal_task
                timings["rebuttal_gap"] = time.perf_counter() - fire_t
                r_start = timings.get("rebuttal_start", bark_t)
                r_end = timings.get("rebuttal_end", bark_end)
                overlap = max(0.0, min(bark_end, r_end) - max(bark_t, r_start))
                timings["bark_rebuttal_overlap"] = overlap
                log(f"[REBUTTAL] {rebuttal}")
                emit("heckle", {"rebuttal": rebuttal, "claim": verdict.get("claim") or ""})
                play_t = time.perf_counter()
                await loop.run_in_executor(None, play_audio, wav)
                timings["rebuttal_play"] = time.perf_counter() - play_t
            finally:
                while not q.empty():  # discard audio captured while agent was talking
                    q.get_nowait()
                speaking = False
            bs_meter = 0.0  # reset the visual after interrupting
            emit("meter", {"value": 0.0})
            log_timings(timings)
            emit("latency", timings)
        else:
            log_timings(timings)
            emit("latency", timings)

    async with client.stt_realtime(
        model_name="default",
        input_format="pcm",
        json_config={"language": "en", "delay_in_frames": 16},
    ) as stt:
        # Keyed by start_s so a revised segment replaces (not duplicates) its prior
        # version. Joined in start_s order to build the turn text.
        segments: dict[float, str] = {}
        high_vad_steps = 0
        flush_pending = False
        flush_id = 0
        last_checked_len = 0  # word-count at last live-check dispatch

        def turn_text() -> str:
            return " ".join(segments[k] for k in sorted(segments)).strip()

        sent = 0

        async def producer():
            nonlocal sent
            while True:
                # Timeout makes the blocking get interruptible so Ctrl+C / cancellation
                # is noticed promptly instead of hanging on a dead executor thread.
                try:
                    chunk = await loop.run_in_executor(None, q.get, True, 0.25)
                except queue.Empty:
                    continue
                if speaking:
                    continue  # don't feed the agent's own voice back into STT
                try:
                    await stt.send_audio(chunk)
                    sent += 1
                    if DEBUG and sent % 25 == 0:
                        log(f"[dbg] sent {sent} chunks, q={q.qsize()} speaking={speaking}")
                except Exception as exc:  # socket closed mid-session
                    log(f"[stt] send failed, stopping producer: {exc}")
                    return

        async def consumer():
            nonlocal high_vad_steps, segments, flush_pending, flush_id, last_checked_len
            async for msg in stt:
                mtype = msg.get("type")
                if DEBUG and mtype != "step":
                    log(f"[dbg] msg={mtype} flush_pending={flush_pending}")
                if mtype == "text":
                    # Revision of the same span (same start_s) replaces the prior text.
                    segments[msg.get("start_s", len(segments))] = msg["text"]
                    text = turn_text()
                    print(f"\r… {text}", end="", flush=True)
                    emit("transcript", {"text": text, "final": False})
                    word_count = len(text.split())
                    if (not speaking
                            and word_count - last_checked_len >= LIVE_CHECK_WORDS
                            and word_count >= LIVE_CHECK_MIN_WORDS):
                        last_checked_len = word_count
                        log(f"[live-check] dispatching at {word_count} words")
                        asyncio.create_task(fact_check_turn(text))
                elif mtype == "step" and msg.get("vad"):
                    inactivity = msg["vad"][-1]["inactivity_prob"]
                    high_vad_steps = high_vad_steps + 1 if inactivity > VAD_THRESHOLD else 0
                    # Fire once per turn: latch until the matching `flushed` returns,
                    # otherwise we spam send_flush every 80 ms of silence and desync.
                    if high_vad_steps >= VAD_STEPS_TO_END and segments and not flush_pending:
                        flush_pending = True
                        flush_id += 1
                        await stt.send_flush(flush_id=flush_id)
                elif mtype == "flushed":
                    text = turn_text()
                    segments = {}
                    high_vad_steps = 0
                    flush_pending = False
                    last_checked_len = 0
                    if text:
                        print(f"\r[TURN] {text}")
                        emit("transcript", {"text": text, "final": True})
                        asyncio.create_task(fact_check_turn(text))
                elif mtype == "end_of_stream":
                    return

        try:
            stream.start()
            await asyncio.gather(producer(), consumer())
        except asyncio.CancelledError:
            pass
        finally:
            stream.stop()
            stream.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
