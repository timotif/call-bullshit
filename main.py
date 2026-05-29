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

from factcheck import (
    check_turn,
    generate_rebuttal,
    open_tts_stream,
    play_audio,
    play_audio_stream,
    warm_up_verdict_path,
)

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
# Hold the first N streamed PCM chunks before playback starts, so a slow/cold
# TTS round can't underrun the device the moment playback outpaces synthesis.
# The barker plays during synthesis and is itself a natural prebuffer, so a
# small N is enough; this only covers the cold/slow case the barker tail misses.
TTS_PREBUFFER_CHUNKS = int(os.environ.get("TTS_PREBUFFER_CHUNKS", "2"))
# Hard ceiling on how long the drainer will block waiting for the next PCM chunk.
# If the background pump dies WITHOUT posting the _REBUTTAL_DONE sentinel (e.g. it
# was killed hard), chunks() would otherwise block its executor thread forever and
# leak a thread-pool slot; repeated occurrences stall the event loop. A rebuttal is
# only a few seconds of audio, so a generous timeout never trips in normal use.
MAX_REBUTTAL_SECONDS = float(os.environ.get("MAX_REBUTTAL_SECONDS", "30"))
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


# Sentinels pushed onto a RebuttalStream's buffer queue to mark stream end /
# stream failure — distinct objects so they can't collide with real PCM bytes.
_REBUTTAL_DONE = object()


class RebuttalStream:
    """A rebuttal's synthesized PCM, pumped off the TTS stream into a buffer.

    Streaming TTS' win is time-to-first-chunk: playback can start on chunk one
    instead of after the whole WAV. But playback is blocking I/O that must run in
    an executor, while the TTS stream is an async generator on the event loop —
    so prepare_rebuttal pumps `speak_stream` chunks into a thread-safe Queue
    (the `_pump` task) and hands back this object. The drainer (in the executor)
    calls `chunks()`, which blocks on that queue. This lets chunks BUFFER during
    barker playback (the pump runs concurrently) and then drain to the device
    once the barker finishes — without the drainer ever touching the event loop.

    `text` is the spoken rebuttal; `sample_rate` is Gradium's output rate (for
    the output device). A mid-stream TTS failure is captured and re-raised out of
    `chunks()` so the turn can stop cleanly (the drainer closes the device); the
    chunks pulled before the failure still play.
    """

    def __init__(self, text: str, sample_rate: int):
        self.text = text
        self.sample_rate = sample_rate
        # Unbounded: a rebuttal is short (~80 tokens) and the barker cover means
        # the buffer fills faster than it drains, so bounding would only risk
        # stalling the pump task on the event loop. Memory is a few KB of PCM.
        self._q: "queue.Queue" = queue.Queue()
        # Set by prepare_rebuttal once the background pump is created. Held so the
        # owner (fact_check_turn's finally) can cancel it on any exit — otherwise a
        # cancelled turn orphans the pump and leaks the Gradium socket (C1).
        self._pump_task: "asyncio.Task | None" = None
        # The TTS objects the pump iterates; closed by cancel_pump() so cancelling
        # the task also releases the underlying stream/socket, not just the coro.
        self._chunk_iter = None  # async generator from stream.iter_bytes()
        self._tts_stream = None  # the gradium TTSStream

    def _put(self, item) -> None:
        self._q.put(item)

    async def cancel_pump(self) -> None:
        """Cancel the background pump AND close the underlying TTS stream.

        Idempotent. Cancelling only the task would leave Gradium's `chunk_iter`
        / `stream` (and their socket) open and could hang the drainer; we
        therefore cancel the task, await its unwind, then aclose the async
        iterator and the stream. Safe to call from a `finally` on any exit path
        (including asyncio.CancelledError) — see fact_check_turn (C1/C2)."""
        task = self._pump_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass  # the pump's own finally already posted the sentinel
        await self._close_tts()

    async def _close_tts(self) -> None:
        """Close the TTS chunk iterator and underlying stream (idempotent)."""
        for obj in (self._chunk_iter, self._tts_stream):
            if obj is None:
                continue
            # Prefer async close (these are async generators / async resources);
            # fall back to sync close. Swallow errors — closing is best-effort.
            # Catch BaseException, not Exception: this runs from fact_check_turn's
            # finally, where a second cancellation can be injected at `await
            # aclose()`. If that CancelledError escaped, _close_tts would unwind
            # before `speaking` is reset, leaving the agent deaf forever (C1/the
            # re-review's residual gap). Closing must complete regardless.
            try:
                aclose = getattr(obj, "aclose", None)
                if aclose is not None:
                    await aclose()
                    continue
                close = getattr(obj, "close", None)
                if close is not None:
                    close()
            except BaseException:
                pass
        self._chunk_iter = None
        self._tts_stream = None

    def chunks(self):
        """Yield buffered PCM chunks in order; blocks until each is available.

        Sync generator, meant to run in an executor (it blocks on the queue and
        ultimately on the output device). Re-raises any TTS-stream failure after
        yielding the chunks that arrived before it, so the drainer's `with`
        closes the device and the caller can reset `speaking`.

        Each get() is bounded by MAX_REBUTTAL_SECONDS: if the pump dies without
        posting the _REBUTTAL_DONE sentinel, we stop draining (rather than block
        the executor thread forever and leak a pool slot — C4).
        """
        while True:
            try:
                item = self._q.get(timeout=MAX_REBUTTAL_SECONDS)
            except queue.Empty:
                log(f"[rebuttal] drainer timed out after {MAX_REBUTTAL_SECONDS:.0f}s "
                    "waiting for audio (pump likely died); stopping drain")
                return
            if item is _REBUTTAL_DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


async def _aclose_quietly(obj) -> None:
    """Best-effort close of a TTS async generator / stream. Swallows errors.

    Prefers async `aclose()` (these are async generators); falls back to a sync
    `close()`. Used on the early-exit paths in prepare_rebuttal (empty stream /
    cancellation before the pump exists) where there is no RebuttalStream yet."""
    if obj is None:
        return
    try:
        aclose = getattr(obj, "aclose", None)
        if aclose is not None:
            await aclose()
            return
        close = getattr(obj, "close", None)
        if close is not None:
            close()
    except Exception:
        pass


async def prepare_rebuttal(verdict: dict, timings: dict | None = None, voice_id: str | None = None, barker_text: str | None = None) -> RebuttalStream:
    """Generate rebuttal text (Nebius) and START streaming speech (Gradium TTS).

    No playback here — returns a RebuttalStream so the PCM can buffer during the
    barker and be drained to the device after it. A background `_pump` task feeds
    the stream's queue as chunks arrive, so synthesis overlaps barker playback.

    Timing keys (only written when `timings` is not None — the calibration
    warm-up call passes None and skips all of them):
      - `rebuttal_start` / `rebuttal_end`: absolute marks bounding prep work.
      - `rebuttal_gen`: Nebius rebuttal-text generation time.
      - `tts` / `tts_first_chunk`: TIME-TO-FIRST-CHUNK (NOT full synthesis).
        Streaming makes "first audible" the metric that sizes the barker; `tts`
        now means that, and `tts_first_chunk` names it explicitly. We do NOT
        block here for the whole stream — `rebuttal_end` is stamped once the
        first chunk lands (prep is "done" the moment we can start playing).

    `voice_id` matches the barker's voice; `barker_text` lets the LLM continue
    the opener seamlessly.
    """
    loop = asyncio.get_running_loop()
    if timings is not None:
        timings["rebuttal_start"] = time.perf_counter()  # absolute: when prep work began
    t0 = time.perf_counter()
    rebuttal = await loop.run_in_executor(None, generate_rebuttal, verdict, barker_text)
    if timings is not None:
        timings["rebuttal_gen"] = time.perf_counter() - t0

    # Open the stream and read its first chunk here so we can (a) learn the
    # device sample_rate from the TTSStream and (b) measure time-to-first-chunk
    # — the metric that now sizes the barker. The rest drains in the background.
    t1 = time.perf_counter()
    stream = await open_tts_stream(rebuttal, voice_id=voice_id) if voice_id else await open_tts_stream(rebuttal)
    chunk_iter = stream.iter_bytes()
    # Read the first chunk here to (a) learn the device rate and (b) mark TTFC.
    # If cancellation lands here, or Gradium returns zero chunks, the stream/
    # chunk_iter are already open and MUST be closed before we leave (C1/C3).
    try:
        first_chunk = await chunk_iter.__anext__()  # blocks until first audio: the TTFC mark
    except StopAsyncIteration:
        # Gradium produced NO audio — a real failure mode (ADR 0002: reasoning
        # models returned empty). Treat as 'no audio': close the stream and hand
        # back an empty RebuttalStream so the turn skips playback gracefully
        # instead of letting StopAsyncIteration escape and crash the consumer.
        log("[rebuttal] TTS returned no audio chunks; aborting rebuttal playback")
        await _aclose_quietly(chunk_iter)
        await _aclose_quietly(stream)
        sample_rate = stream.sample_rate or SAMPLE_RATE
        rs = RebuttalStream(rebuttal, sample_rate)
        rs._put(_REBUTTAL_DONE)  # chunks() drains to nothing immediately
        if timings is not None:
            ttfc = time.perf_counter() - t1
            timings["tts_first_chunk"] = ttfc
            timings["tts"] = ttfc
            timings["rebuttal_end"] = time.perf_counter()
        return rs
    except BaseException:
        # Cancellation (Ctrl+C) before the pump exists: the only open resources
        # are the stream/chunk_iter we just created — close them, then re-raise.
        await _aclose_quietly(chunk_iter)
        await _aclose_quietly(stream)
        raise

    if timings is not None:
        ttfc = time.perf_counter() - t1
        timings["tts_first_chunk"] = ttfc
        timings["tts"] = ttfc  # `tts` now means time-to-first-chunk (streaming)
        timings["rebuttal_end"] = time.perf_counter()  # prep "done": ready to play

    # Gradium reports the synthesized rate on the stream; fall back to the
    # project's known pcm rate if the SDK leaves it unset.
    sample_rate = stream.sample_rate or SAMPLE_RATE
    rs = RebuttalStream(rebuttal, sample_rate)
    rs._chunk_iter = chunk_iter  # held so cancel_pump() can close the socket (C1)
    rs._tts_stream = stream
    rs._put(first_chunk)

    async def _pump() -> None:
        """Drain the rest of the TTS stream into the buffer queue, then signal end.

        A TTS stream can raise AFTER some audio has played; we capture that
        exception onto the queue so `chunks()` re-raises it in the drainer
        (which then closes the device) instead of crashing the event loop.
        """
        try:
            async for chunk in chunk_iter:
                rs._put(chunk)
        except Exception as exc:  # mid-stream synthesis failure (NOT cancellation)
            rs._put(exc)
        # NOTE: we deliberately do NOT catch CancelledError/BaseException here —
        # cancellation (from cancel_pump) must propagate so the task is marked
        # cancelled and unwinds. The `finally` still posts the sentinel so any
        # blocked drainer is released even on cancellation.
        finally:
            rs._put(_REBUTTAL_DONE)

    rs._pump_task = loop.create_task(_pump(), name="rebuttal-pump")
    return rs


async def _drain_for_calibration(rs: RebuttalStream) -> None:
    """Fully consume a calibration RebuttalStream WITHOUT touching the audio device.

    The background pump only finishes (and the TTS stream only closes) once its
    chunks are consumed; if we left them buffered, the pump task would dangle and
    the socket would stay open. We drain in a thread so the blocking queue reads
    don't stall the event loop, and discard the bytes — calibration measures
    latency, it does not play audio.
    """
    loop = asyncio.get_running_loop()
    def _consume() -> None:
        for _ in rs.chunks():
            pass
    await loop.run_in_executor(None, _consume)


async def calibrate_prep_latency() -> float:
    """Mock two rebuttals at startup to size barkers for this session.

    The FIRST round is a throwaway warm-up: the rebuttal model and Gradium TTS
    both pay a cold-start on their first request after idle, so measuring a cold
    round would over-estimate the budget and pick barkers longer than real fires
    need. We discard the first round and calibrate on the SECOND (warm) one, so
    the budget reflects steady-state and CALIBRATION_MARGIN only covers genuine
    per-call variance.

    KEY CHANGE for streaming TTS: with buffered TTS the barker had to cover
    `gen + full synthesis` (~4.3s warm), which over-sized barkers. Streaming
    starts playing on the first chunk, so the barker now only has to cover
    `gen + time-to-first-chunk` (~1.2s warm). prepare_rebuttal records `tts` as
    time-to-first-chunk (not full synthesis), so `gen + tts` here is exactly that
    smaller budget — this is what shrinks the barker and stops the rambling. We
    still fully drain each mock stream so the warm-up actually exercises (and
    closes) the TTS path; the drain time is NOT part of the measured budget.

    Returns the prep budget in seconds: warm (gen + time-to-first-chunk) + margin.
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
        rs_cold = await prepare_rebuttal(mock_verdict)          # cold throwaway: warms gen + TTS
        await _drain_for_calibration(rs_cold)                   # close the cold stream
        rs_warm = await prepare_rebuttal(mock_verdict, timings)  # warm: this is what we measure
        await _drain_for_calibration(rs_warm)                   # close the warm stream
    except Exception as exc:
        print(f"[WARN] calibration failed ({exc}); using longest barker", flush=True)
        emit("status", {"phase": "calibration_failed", "text": f"Calibration failed: {exc}"})
        return 0.0
    # `tts` is now time-to-first-chunk, so prep = gen + first-audio (not full TTS).
    prep = timings.get("rebuttal_gen", 0.0) + timings.get("tts", 0.0)
    budget = min(prep + CALIBRATION_MARGIN, CALIBRATION_MAX)
    capped = " (capped)" if prep + CALIBRATION_MARGIN > CALIBRATION_MAX else ""
    print(
        f"Calibrated: gen+first-chunk={prep:.1f}s, barker budget={budget:.1f}s "
        f"(+{CALIBRATION_MARGIN:.1f}s margin{capped})\n",
        flush=True,
    )
    emit("status", {
        "phase": "ready",
        "text": f"Ready — gen+first-chunk={prep:.1f}s, barker budget={budget:.1f}s{capped}",
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
            rs: "RebuttalStream | None" = None
            rebuttal_task: "asyncio.Task | None" = None
            try:
                fire_t = time.perf_counter()
                # Start streaming TTS concurrently with the barker. prepare_rebuttal
                # returns once the FIRST chunk lands; its background pump keeps
                # buffering the rest while the barker plays — so by barker-end the
                # rebuttal audio is (mostly) ready to drain with no buffering wait.
                rebuttal_task = asyncio.create_task(
                    prepare_rebuttal(verdict, timings, voice_id=rebuttal_voice_id, barker_text=rebuttal_barker_text)
                )
                bark_t = time.perf_counter()
                # If play_barker raises/cancels, the finally below cancels
                # rebuttal_task (and its pump) so neither is orphaned (C2).
                await loop.run_in_executor(None, play_barker, barkers, prep_budget, chosen_barker)
                bark_end = time.perf_counter()
                timings["barker_play"] = bark_end - bark_t
                rs = await rebuttal_task
                rebuttal_task = None  # completed: ownership moves to rs (cancel_pump)
                timings["rebuttal_gap"] = time.perf_counter() - fire_t
                r_start = timings.get("rebuttal_start", bark_t)
                r_end = timings.get("rebuttal_end", bark_end)
                overlap = max(0.0, min(bark_end, r_end) - max(bark_t, r_start))
                timings["bark_rebuttal_overlap"] = overlap
                log(f"[REBUTTAL] {rs.text}")
                emit("heckle", {"rebuttal": rs.text, "claim": verdict.get("claim") or ""})
                play_t = time.perf_counter()
                # Drain buffered PCM chunks to the device in the executor (blocking
                # I/O — keep it off the event loop so the STT producer/consumer keep
                # running). speaking stays True across the WHOLE drain: rs.chunks()
                # blocks until the last chunk is written, and only then does the
                # finally below flip it back, so the agent's own voice isn't fed
                # back into STT. A mid-stream TTS failure re-raises out of
                # play_audio_stream; the finally still drains the mic queue and
                # resets speaking, so the turn ends cleanly instead of crashing.
                try:
                    await loop.run_in_executor(
                        None, play_audio_stream, rs.chunks(), rs.sample_rate, 1, TTS_PREBUFFER_CHUNKS,
                    )
                except Exception as exc:
                    log(f"[rebuttal] playback stopped early: {exc}")
                timings["rebuttal_play"] = time.perf_counter() - play_t
            except Exception as exc:
                # No rebuttal-path failure may crash the turn consumer (C3). Log
                # and continue; the finally still cleans up. (CancelledError is a
                # BaseException, so Ctrl+C still propagates through here.)
                log(f"[rebuttal] aborted: {exc}")
            finally:
                # Cancel the prep task if play_barker (or anything before the
                # await) failed and left it running, so its _pump isn't orphaned (C2).
                if rebuttal_task is not None and not rebuttal_task.done():
                    rebuttal_task.cancel()
                    try:
                        await rebuttal_task
                    except (asyncio.CancelledError, Exception):
                        pass
                # If prep completed, cancel its background pump and close the
                # Gradium stream/socket — covers normal exit AND cancellation (C1).
                if rs is not None:
                    await rs.cancel_pump()
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
