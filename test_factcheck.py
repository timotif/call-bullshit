"""TDD tests for the pure functions in factcheck.py.

No network calls — only parse_claim_response, score_verdict,
parse_verdict_response, and _fallback_rebuttal are tested.
"""
import pytest
from factcheck import parse_claim_response, score_verdict, parse_verdict_response, _fallback_rebuttal


# ---------------------------------------------------------------------------
# parse_claim_response
# ---------------------------------------------------------------------------

def test_parse_plain_claim():
    assert parse_claim_response("The Great Wall is visible from space.") == \
        "The Great Wall is visible from space."


def test_parse_quoted_claim():
    result = parse_claim_response('"Vaccines cause autism."')
    assert result == "Vaccines cause autism."


def test_parse_none_literal():
    assert parse_claim_response("NONE") is None


def test_parse_none_lowercase_with_punct():
    assert parse_claim_response("none.") is None


def test_parse_empty_string():
    assert parse_claim_response("") is None


def test_parse_whitespace_only():
    assert parse_claim_response("   ") is None


def test_parse_leading_claim_label():
    result = parse_claim_response("Claim: The moon is made of cheese.")
    assert result == "The moon is made of cheese."


def test_parse_strips_surrounding_whitespace():
    result = parse_claim_response("  Lightning never strikes the same place twice.  ")
    assert result == "Lightning never strikes the same place twice."


# ---------------------------------------------------------------------------
# score_verdict
# ---------------------------------------------------------------------------

RESULTS_WITH_SCORE = [
    {"title": "Science Daily", "url": "https://example.com/a", "content": "...", "score": 0.9},
    {"title": "Wikipedia", "url": "https://example.com/b", "content": "...", "score": 0.7},
]

def test_score_contradicted_cue():
    verdict = score_verdict(
        "The Great Wall is visible from space.",
        "This is false. Astronauts cannot see the Great Wall from orbit.",
        RESULTS_WITH_SCORE,
    )
    assert verdict["status"] == "contradicted"
    assert verdict["confidence"] > 0.5
    assert verdict["source_url"] == "https://example.com/a"
    assert verdict["source_title"] == "Science Daily"
    assert verdict["claim"] == "The Great Wall is visible from space."


def test_score_supported_cue():
    verdict = score_verdict(
        "The Earth orbits the Sun.",
        "This is correct. The Earth indeed orbits the Sun.",
        RESULTS_WITH_SCORE,
    )
    assert verdict["status"] == "supported"
    assert verdict["confidence"] > 0.5


def test_score_ambiguous_gives_mixed():
    verdict = score_verdict(
        "Coffee cures cancer.",
        "Some studies suggest benefits while others disagree. Evidence is mixed.",
        RESULTS_WITH_SCORE,
    )
    assert verdict["status"] == "mixed"


def test_score_empty_results_gives_unknown():
    verdict = score_verdict(
        "Bigfoot is real.",
        "",
        [],
    )
    assert verdict["status"] == "unknown"
    assert verdict["source_url"] is None
    assert verdict["source_title"] is None


def test_score_picks_highest_score_result():
    results = [
        {"title": "Low", "url": "https://low.com", "content": ".", "score": 0.3},
        {"title": "High", "url": "https://high.com", "content": ".", "score": 0.95},
    ]
    verdict = score_verdict("some claim", "It is supported by evidence.", results)
    assert verdict["source_url"] == "https://high.com"
    assert verdict["source_title"] == "High"


def test_score_verdict_has_required_keys():
    verdict = score_verdict("test claim", "true", RESULTS_WITH_SCORE)
    for key in ("claim", "status", "confidence", "source_url", "source_title", "summary"):
        assert key in verdict


def test_score_confidence_in_range():
    verdict = score_verdict("test claim", "false contradiction incorrect", RESULTS_WITH_SCORE)
    assert 0.0 <= verdict["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# parse_verdict_response
# ---------------------------------------------------------------------------

_CLAIM = "The Great Wall is visible from space."
_ANSWER = "The Great Wall is not visible from space with the naked eye."
_RESULTS = [
    {"title": "NASA", "url": "https://nasa.gov/wall", "content": "...", "score": 0.95},
    {"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Wall", "content": "...", "score": 0.7},
]


def test_pvr_valid_json_contradicted():
    v = parse_verdict_response('{"status": "contradicted", "confidence": 0.95}', _CLAIM, _ANSWER, _RESULTS)
    assert v["status"] == "contradicted"
    assert v["confidence"] == pytest.approx(0.95)
    assert v["claim"] == _CLAIM
    assert v["source_url"] == "https://nasa.gov/wall"
    assert v["source_title"] == "NASA"
    assert _ANSWER[:300] in v["summary"]


def test_pvr_json_in_markdown_fences():
    raw = "```json\n{\"status\": \"contradicted\", \"confidence\": 0.9}\n```"
    v = parse_verdict_response(raw, _CLAIM, _ANSWER, _RESULTS)
    assert v["status"] == "contradicted"
    assert v["confidence"] == pytest.approx(0.9)


def test_pvr_json_with_trailing_prose():
    raw = '{"status": "supported", "confidence": 0.8} This is my reasoning.'
    v = parse_verdict_response(raw, _CLAIM, _ANSWER, _RESULTS)
    assert v["status"] == "supported"
    assert v["confidence"] == pytest.approx(0.8)


def test_pvr_garbage_input_returns_unknown():
    v = parse_verdict_response("I cannot determine anything.", _CLAIM, _ANSWER, _RESULTS)
    assert v["status"] == "unknown"
    assert v["confidence"] == pytest.approx(0.0)


def test_pvr_missing_confidence_defaults_to_zero():
    raw = '{"status": "mixed"}'
    v = parse_verdict_response(raw, _CLAIM, _ANSWER, _RESULTS)
    assert v["status"] == "mixed"
    assert v["confidence"] == pytest.approx(0.0)


def test_pvr_status_normalized_to_lowercase():
    raw = '{"status": "Contradicted", "confidence": 0.88}'
    v = parse_verdict_response(raw, _CLAIM, _ANSWER, _RESULTS)
    assert v["status"] == "contradicted"


def test_pvr_has_required_keys():
    v = parse_verdict_response('{"status": "unknown", "confidence": 0.0}', _CLAIM, _ANSWER, _RESULTS)
    for key in ("claim", "status", "confidence", "source_url", "source_title", "summary"):
        assert key in v


def test_pvr_empty_results_gives_none_source():
    v = parse_verdict_response('{"status": "unknown", "confidence": 0.0}', _CLAIM, _ANSWER, [])
    assert v["source_url"] is None
    assert v["source_title"] is None


# ---------------------------------------------------------------------------
# _fallback_rebuttal (pure helper — slice 4)
# ---------------------------------------------------------------------------

def test_fallback_rebuttal_normal_summary():
    result = _fallback_rebuttal(
        "The Great Wall is not visible from space with the naked eye. Astronauts confirmed this."
    )
    assert result.startswith("Actually, that's not right.")
    assert "Great Wall" in result


def test_fallback_rebuttal_empty_summary():
    assert _fallback_rebuttal("") == "Actually, that's not right."


def test_fallback_rebuttal_no_answer_available():
    assert _fallback_rebuttal("No answer available.") == "Actually, that's not right."


def test_fallback_rebuttal_single_sentence_no_dot():
    result = _fallback_rebuttal("Vaccines do not cause autism")
    assert "Actually, that's not right." in result


def test_fallback_rebuttal_ends_with_period():
    result = _fallback_rebuttal("The moon is not made of cheese. It is made of rock.")
    assert result.endswith(".")


# ---------------------------------------------------------------------------
# Unresolved-subject / vague-claim guard
#
# These test the layers that can catch vague claims WITHOUT a live LLM call:
#   1. parse_claim_response — the LLM should return "NONE" for unresolvable
#      subjects; parse_claim_response must honour that.
#   2. score_verdict (heuristic fallback) — if a vague claim somehow reaches
#      this layer and Tavily returns no useful answer, it must not fire
#      "contradicted"; it should return "unknown" or at most "mixed".
#   3. parse_verdict_response — if the judge LLM is correctly prompted it
#      returns {"status":"unknown"} for these cases; verify that is preserved.
#
# The representative turn is the full pitch excerpt from the user report:
#   "our country has reduced unemployment … invested more in scientific
#    research than at any moment since the 1980s … funded schools …
#    planted millions of trees across the nation."
# Every claim in that turn has "our country / our nation" as subject — no
# named referent → none should fire.
# ---------------------------------------------------------------------------

_VAGUE_TURNS = [
    # The exact claim that triggered the false positive
    "our nation invested in research more than any time in the past",
    # Full pitch excerpt (multi-claim; extractor should return NONE or a named-entity claim)
    (
        "Today we stand at a turning point. In the last five years, our country has "
        "reduced unemployment, expanded public transport, and invested more in scientific "
        "research than at any moment since the 1980s. We reopened libraries, funded "
        "schools, and planted millions of trees across the nation."
    ),
    # Other unresolvable-subject forms
    "we funded schools",
    "our government reduced the deficit last year",
    "this country leads the world in renewable energy",
    "my team shipped more features than any quarter before",
]

_NAMED_CLAIMS = [
    "The Great Wall of China is visible from space with the naked eye.",
    "France is the most visited country in the world.",
    "Apple was founded in 1976.",
]



@pytest.mark.parametrize("claim", _VAGUE_TURNS)
def test_score_verdict_vague_subject_no_evidence_is_unknown(claim):
    """Heuristic fallback: empty Tavily answer + no results → unknown, not contradicted."""
    v = score_verdict(claim, "", [])
    assert v["status"] == "unknown"
    assert v["confidence"] == pytest.approx(0.0)


@pytest.mark.parametrize("claim", _VAGUE_TURNS)
def test_score_verdict_vague_subject_off_topic_answer_not_contradicted(claim):
    """Heuristic fallback: off-topic answer with no contradiction cue words
    must not produce 'contradicted' for a vague claim."""
    off_topic = "Here is some general information about economic trends."
    v = score_verdict(claim, off_topic, [])
    assert v["status"] != "contradicted"


@pytest.mark.parametrize("claim", _VAGUE_TURNS)
def test_parse_verdict_response_unknown_preserved_for_vague(claim):
    """If the judge correctly returns unknown for a vague claim,
    parse_verdict_response must preserve it, not upgrade it."""
    raw = '{"status": "unknown", "confidence": 0.0}'
    v = parse_verdict_response(raw, claim, "", [])
    assert v["status"] == "unknown"


@pytest.mark.parametrize("claim", _NAMED_CLAIMS)
def test_score_verdict_named_entity_can_still_fire(claim):
    """Named-entity claims with clear contradicting text should still reach
    'contradicted' — the vague-subject guard must not over-suppress."""
    answer = "This claim is false. Evidence directly contradicts this assertion."
    v = score_verdict(claim, answer, [])
    assert v["status"] == "contradicted"


# ---------------------------------------------------------------------------
# parse_claim_response — vague subject (fixed test)
# ---------------------------------------------------------------------------
# The LLM is responsible for detecting vague/unresolvable subjects and
# returning "NONE". parse_claim_response's role is only to translate "NONE"
# into Python None. If the LLM echoes back a vague string directly,
# parse_claim_response passes it through — it has no knowledge of whether a
# subject is vague. The guard lives in the LLM prompt, not here.

def test_parse_claim_response_passes_through_vague_strings():
    """parse_claim_response does NOT filter vague subjects — that is the LLM's
    responsibility. A vague string that the LLM fails to suppress is passed
    through as a valid claim string."""
    vague = "our nation invested in research more than any time in the past"
    assert parse_claim_response(vague) == vague


# ---------------------------------------------------------------------------
# play_audio — stream lifecycle (resource safety)
# ---------------------------------------------------------------------------

import io
import wave
import struct
from unittest.mock import MagicMock, patch, call


def _make_wav_bytes(sr: int = 24000, channels: int = 1, n_frames: int = 100) -> bytes:
    """Build a minimal valid WAV (PCM s16le) in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def test_play_audio_closes_stream_on_success():
    """play_audio must close the audio stream after successful playback."""
    from factcheck import play_audio

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    with patch("sounddevice.RawOutputStream", return_value=mock_stream):
        play_audio(_make_wav_bytes())

    mock_stream.__exit__.assert_called_once()


def test_play_audio_closes_stream_on_write_error():
    """play_audio must close the audio stream even when write() raises."""
    from factcheck import play_audio

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.write.side_effect = RuntimeError("buffer overrun")

    with patch("sounddevice.RawOutputStream", return_value=mock_stream):
        with pytest.raises(RuntimeError, match="buffer overrun"):
            play_audio(_make_wav_bytes())

    mock_stream.__exit__.assert_called_once()
