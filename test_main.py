"""Tests for pure logic in main.py — no audio hardware, no network."""
import sys
import os

# Stub heavy optional deps before importing main
import unittest.mock as mock
sys.modules.setdefault("sounddevice", mock.MagicMock())
sys.modules.setdefault("gradium", mock.MagicMock())
sys.modules.setdefault("aiohttp", mock.MagicMock())
sys.modules.setdefault("aiohttp.web", mock.MagicMock())

import pytest
import main
from main import pick_barker


@pytest.fixture(autouse=True)
def reset_barker_state():
    """Clear shuffle-queue state between tests so each test starts fresh."""
    main._barker_queues.clear()
    main._barker_last.clear()
    yield
    main._barker_queues.clear()
    main._barker_last.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _barkers(durations: list[float]) -> list[dict]:
    """Build a minimal barker list with given durations."""
    return [
        {"file": f"barker_{i:02d}.wav", "text": f"text {i}",
         "voice": f"voice{i}", "voice_id": f"vid{i}", "duration": d}
        for i, d in enumerate(durations)
    ]


# ---------------------------------------------------------------------------
# Existing contract: budget coverage still works
# ---------------------------------------------------------------------------

def test_pick_barker_returns_covering_barker():
    """Result must have duration >= budget."""
    barkers = _barkers([3.0, 6.0, 10.0])
    pick = pick_barker(barkers, budget=5.0)
    assert pick["duration"] >= 5.0


def test_pick_barker_falls_back_to_longest_when_none_cover():
    """When no barker covers the budget, return the longest."""
    barkers = _barkers([2.0, 4.0])
    pick = pick_barker(barkers, budget=99.0)
    assert pick["duration"] == 4.0


# ---------------------------------------------------------------------------
# Randomization: no two consecutive identical picks
# ---------------------------------------------------------------------------

def test_pick_barker_does_not_repeat_consecutively():
    """Calling pick_barker twice in a row must not return the same barker
    when there are multiple eligible candidates."""
    barkers = _barkers([6.0, 7.0, 8.0])
    first = pick_barker(barkers, budget=5.0)
    second = pick_barker(barkers, budget=5.0)
    assert first["file"] != second["file"]


def test_pick_barker_cycles_through_full_pool_before_repeating():
    """With N eligible barkers, the same barker must not appear twice
    until all others in the eligible set have been played."""
    barkers = _barkers([5.0, 6.0, 7.0])
    seen = []
    for _ in range(len(barkers)):
        pick = pick_barker(barkers, budget=4.0)
        seen.append(pick["file"])
    # All three should be distinct in one full cycle
    assert len(set(seen)) == len(barkers)


def test_pick_barker_exhausted_pool_resets_and_continues():
    """After exhausting the pool the sequence resets — next pick is valid."""
    barkers = _barkers([5.0, 6.0])
    seen = []
    for _ in range(4):  # two full cycles
        pick = pick_barker(barkers, budget=4.0)
        seen.append(pick["file"])
    # No two adjacent picks are the same
    for a, b in zip(seen, seen[1:]):
        assert a != b


def test_play_barker_uses_chosen_without_extra_pick():
    """play_barker(chosen=X) must play X and not consume another slot from the
    shuffle queue — prevents the double-pick that mismatches rebuttal voice."""
    from unittest.mock import patch, MagicMock
    barkers = _barkers([6.0, 7.0, 8.0])
    chosen = barkers[1]  # pick slot 1 explicitly

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    with patch("main.BARKERS_DIR") as mock_dir, \
         patch("main.sd") as mock_sd, \
         patch("main.wave") as mock_wave:
        mock_sd.RawOutputStream.return_value = mock_stream
        mock_path = MagicMock()
        mock_path.__str__ = lambda s: "barker_01.wav"
        mock_path.read_bytes.return_value = b"\x00" * (44 + 100)
        mock_dir.__truediv__ = lambda s, f: mock_path
        mock_wf = MagicMock()
        mock_wf.__enter__ = MagicMock(return_value=mock_wf)
        mock_wf.__exit__ = MagicMock(return_value=False)
        mock_wf.getframerate.return_value = 24000
        mock_wf.getnchannels.return_value = 1
        mock_wave.open.return_value = mock_wf

        result = main.play_barker(barkers, budget=5.0, chosen=chosen)

    # Must return exactly the chosen barker
    assert result is chosen
    # Queue must still be untouched (no extra pick_barker call)
    assert frozenset(b["file"] for b in barkers) not in main._barker_queues


def test_pick_barker_single_eligible_always_returns_it():
    """With only one eligible barker, it must still be returned (no infinite loop)."""
    barkers = _barkers([10.0, 2.0])
    # Only the 10s barker covers a budget of 9s
    for _ in range(3):
        pick = pick_barker(barkers, budget=9.0)
        assert pick["duration"] == 10.0


def test_pick_barker_fallback_pool_also_randomizes():
    """When no barker covers the budget, fallback pool is also shuffled —
    no two consecutive picks are the same."""
    barkers = _barkers([2.0, 2.0, 2.0])  # all same duration, all "longest"
    picks = [pick_barker(barkers, budget=99.0) for _ in range(6)]
    files = [p["file"] for p in picks]
    for a, b in zip(files, files[1:]):
        assert a != b


# ---------------------------------------------------------------------------
# prepare_rebuttal — streaming TTS contract (slice: streaming TTS)
# ---------------------------------------------------------------------------
# prepare_rebuttal now starts a streaming TTS and returns a RebuttalStream whose
# .chunks() yields the PCM chunks for the playback drainer. It must:
#   - generate the rebuttal text (Nebius) off-thread,
#   - start speak_stream and record `tts` as TIME-TO-FIRST-CHUNK (not full TTS),
#   - preserve timing keys rebuttal_start / rebuttal_gen / tts / rebuttal_end,
#   - expose the text, the device sample rate, and the ordered chunk stream.
# speak_stream + generate_rebuttal are the external boundaries — patched here.

import asyncio
import factcheck


class _FakeTTSStream:
    def __init__(self, chunks, sample_rate=24000, fail_after=None, block_after=None):
        self._chunks = chunks
        self.sample_rate = sample_rate
        self._fail_after = fail_after
        # If set, iter_bytes() awaits forever after yielding `block_after` chunks
        # (models a stream that stalls mid-synthesis — used to test cancellation).
        self._block_after = block_after
        self.closed = False  # set True if the iter_bytes generator is aclose()'d

    async def iter_bytes(self):
        try:
            for i, c in enumerate(self._chunks):
                if self._fail_after is not None and i >= self._fail_after:
                    raise RuntimeError("tts stream died")
                if self._block_after is not None and i >= self._block_after:
                    await asyncio.Event().wait()  # block until cancelled/closed
                yield c
        finally:
            # GeneratorExit (from aclose) or normal exhaustion both land here.
            self.closed = True


def _fake_speak_stream(chunks, sample_rate=24000, fail_after=None, block_after=None):
    """Build a speak_stream replacement (async generator) over canned chunks.

    Mirrors factcheck.speak_stream: it must expose sample_rate to prepare_rebuttal.
    prepare_rebuttal reads sample_rate via the underlying TTSStream, so we route
    through speak_stream's real shape by patching factcheck._get_gradium_client.
    Returns (fake_client, fake_stream) so tests can assert close behaviour.
    """
    fake_stream = _FakeTTSStream(chunks, sample_rate, fail_after, block_after)
    fake_client = mock.MagicMock()

    async def _tts_stream(setup, text):
        return fake_stream
    fake_client.tts_stream = _tts_stream
    fake_client._fake_stream = fake_stream
    return fake_client


def _run(coro):
    return asyncio.run(coro)


def test_prepare_rebuttal_streams_chunks_in_order():
    """The returned stream yields exactly the PCM chunks Gradium produced, in order."""
    chunks = [b"\x01\x01", b"\x02\x02", b"\x03\x03"]
    fake_client = _fake_speak_stream(chunks)

    with mock.patch.object(main, "generate_rebuttal", return_value="You're wrong."), \
         mock.patch.object(factcheck, "_get_gradium_client", return_value=fake_client):
        rs = _run(main.prepare_rebuttal({"claim": "c", "summary": "s"}))
        drained = list(rs.chunks())

    assert rs.text == "You're wrong."
    assert drained == chunks


def test_prepare_rebuttal_reports_sample_rate():
    """The stream exposes the device sample rate from Gradium (for the drainer)."""
    fake_client = _fake_speak_stream([b"\x00\x00"], sample_rate=24000)
    with mock.patch.object(main, "generate_rebuttal", return_value="x"), \
         mock.patch.object(factcheck, "_get_gradium_client", return_value=fake_client):
        rs = _run(main.prepare_rebuttal({"claim": "c", "summary": "s"}))
        list(rs.chunks())
    assert rs.sample_rate == 24000


def test_prepare_rebuttal_tts_measures_time_to_first_chunk():
    """`tts` timing must measure time-to-first-chunk, not full synthesis — this
    is what calibration keys off to shrink the barker. We also expose a dedicated
    first-chunk key and preserve the gen / start / end timing keys."""
    chunks = [b"\x00\x00", b"\x11\x11"]
    fake_client = _fake_speak_stream(chunks)
    timings = {}
    with mock.patch.object(main, "generate_rebuttal", return_value="x"), \
         mock.patch.object(factcheck, "_get_gradium_client", return_value=fake_client):
        rs = _run(main.prepare_rebuttal({"claim": "c", "summary": "s"}, timings))
        list(rs.chunks())

    for key in ("rebuttal_start", "rebuttal_gen", "tts", "rebuttal_end", "tts_first_chunk"):
        assert key in timings, f"missing timing key {key}"
    # time-to-first-chunk is recorded and non-negative.
    assert timings["tts_first_chunk"] >= 0.0
    assert timings["tts"] == pytest.approx(timings["tts_first_chunk"])


def test_prepare_rebuttal_midstream_failure_propagates_to_drainer():
    """A stream that dies after the first chunk must surface the error to the
    drainer (so the turn can stop cleanly) — the first chunk still drains."""
    fake_client = _fake_speak_stream([b"\x00\x00", b"\x11\x11"], fail_after=1)
    with mock.patch.object(main, "generate_rebuttal", return_value="x"), \
         mock.patch.object(factcheck, "_get_gradium_client", return_value=fake_client):
        rs = _run(main.prepare_rebuttal({"claim": "c", "summary": "s"}))
        got = []
        with pytest.raises(RuntimeError, match="tts stream died"):
            for c in rs.chunks():
                got.append(c)

    assert got == [b"\x00\x00"]  # the one good chunk drained before the failure


# ---------------------------------------------------------------------------
# Cancellation / cleanup contract (review findings C1/C3/C4)
# ---------------------------------------------------------------------------

def test_prepare_rebuttal_empty_tts_stream_aborts_gracefully():
    """T-a: a zero-chunk TTS stream must NOT raise out of the turn path.

    Gradium has been observed returning zero audio chunks (ADR 0002, reasoning
    models). prepare_rebuttal must treat that as 'no audio', close the stream,
    and return a RebuttalStream whose chunks() yields nothing (the turn skips
    playback) instead of letting StopAsyncIteration escape and crash the session.
    """
    fake_client = _fake_speak_stream([])  # zero chunks
    with mock.patch.object(main, "generate_rebuttal", return_value="x"), \
         mock.patch.object(factcheck, "_get_gradium_client", return_value=fake_client):
        rs = _run(main.prepare_rebuttal({"claim": "c", "summary": "s"}))
        assert list(rs.chunks()) == []  # no audio, drains immediately, no raise
    assert fake_client._fake_stream.closed  # the empty stream was closed


def test_prepare_rebuttal_cancel_mid_first_chunk_cleans_up():
    """T-b: cancelling prepare_rebuttal while it blocks on the first chunk must
    close the underlying stream/iterator and leave no orphaned pump task.

    We model a stream that never yields its first chunk (blocks forever). The
    awaiting task is cancelled; prepare_rebuttal must close the chunk iterator
    (so the socket is released) and propagate CancelledError. No pump task is
    created in this case (we never got past the first chunk)."""
    fake_client = _fake_speak_stream([b"\x00\x00"], block_after=0)  # blocks before chunk 0

    async def scenario():
        with mock.patch.object(main, "generate_rebuttal", return_value="x"), \
             mock.patch.object(factcheck, "_get_gradium_client", return_value=fake_client):
            task = asyncio.create_task(
                main.prepare_rebuttal({"claim": "c", "summary": "s"})
            )
            await asyncio.sleep(0.05)  # let it reach the blocking __anext__
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.01)  # allow aclose to finish

    _run(scenario())
    assert fake_client._fake_stream.closed  # stream closed on cancellation


def test_rebuttal_stream_cancel_pump_after_first_chunk():
    """T-b (post-first-chunk): once the pump is running, cancel_pump() must
    cancel the pump task AND close the underlying TTS stream/iterator."""
    fake_client = _fake_speak_stream([b"\x00\x00"], block_after=1)  # 1 chunk then block

    async def scenario():
        with mock.patch.object(main, "generate_rebuttal", return_value="x"), \
             mock.patch.object(factcheck, "_get_gradium_client", return_value=fake_client):
            rs = await main.prepare_rebuttal({"claim": "c", "summary": "s"})
            assert rs._pump_task is not None and not rs._pump_task.done()
            await rs.cancel_pump()
            assert rs._pump_task.cancelled() or rs._pump_task.done()
        return rs

    rs = _run(scenario())
    assert fake_client._fake_stream.closed  # stream closed by cancel_pump


def test_chunks_drainer_unblocks_via_timeout(monkeypatch):
    """T-c: if the pump dies without posting the DONE sentinel, chunks() must
    stop draining via a timeout instead of blocking the executor thread forever."""
    monkeypatch.setattr(main, "MAX_REBUTTAL_SECONDS", 0.05)
    rs = main.RebuttalStream("x", 24000)
    rs._put(b"\x00\x00")  # one chunk, then NOTHING (no sentinel, simulating a dead pump)
    got = list(rs.chunks())  # must return (via timeout), not hang
    assert got == [b"\x00\x00"]
