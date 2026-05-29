"""Fact-checking brain for Call Bullshit.

Pure functions (parse_claim_response, score_verdict, parse_verdict_response,
_fallback_rebuttal) are unit-tested and make no network calls.
Adapter functions (extract_claim, fact_check, check_turn, generate_rebuttal,
speak) call live APIs (Nebius, Tavily, Gradium).
"""

import io
import json
import os
import re
import sys
import time
import wave

from dotenv import load_dotenv

load_dotenv()

WAV_HEADER_BYTES = 44  # canonical RIFF/WAV header size; Gradium WAVs use standard 44-byte headers

# ---------------------------------------------------------------------------
# Lazy-initialised cached API clients (avoid per-call connection churn)
# ---------------------------------------------------------------------------

_nebius_client: "OpenAI | None" = None
_tavily_client: "TavilyClient | None" = None


def _get_nebius_client() -> "OpenAI":
    global _nebius_client
    if _nebius_client is None:
        from openai import OpenAI
        _nebius_client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.environ["NEBIUS_API_KEY"],
        )
    return _nebius_client


def _get_tavily_client() -> "TavilyClient":
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _tavily_client


# ---------------------------------------------------------------------------
# Heuristic word lists for score_verdict
# ---------------------------------------------------------------------------
_CONTRADICTION_CUES = {
    "false", "incorrect", "wrong", "myth", "debunked", "misleading",
    "inaccurate", "untrue", "not true", "cannot", "no evidence",
    "contradiction", "contradicted", "disproven", "disputed",
}
_SUPPORT_CUES = {
    "true", "correct", "confirmed", "accurate", "indeed", "supported",
    "evidence shows", "research shows", "studies show", "verified",
}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def parse_claim_response(raw: str) -> str | None:
    """Parse Nebius's raw reply for the claim-extraction prompt.

    Returns the cleaned claim string, or None if the model said NONE or
    returned nothing useful.
    """
    text = raw.strip().strip('"').strip("'").strip()

    if not text:
        return None

    # Strip a leading "Claim: " label the model sometimes adds.
    if text.lower().startswith("claim:"):
        text = text[len("claim:"):].strip()

    # Model signalled no checkable claim.
    if text.lower().rstrip(".!? ") == "none":
        return None

    return text if text else None


def score_verdict(claim: str, tavily_answer: str, tavily_results: list[dict]) -> dict:
    """Derive a verdict from Tavily's answer string and result list.

    Status heuristic: scan tavily_answer for contradiction vs support cue words.
    Priority: contradicted > supported > mixed > unknown.
    """
    answer_lower = tavily_answer.lower()

    has_contra = any(cue in answer_lower for cue in _CONTRADICTION_CUES)
    has_support = any(cue in answer_lower for cue in _SUPPORT_CUES)

    if has_contra and has_support:
        status = "mixed"
        confidence = 0.5
    elif has_contra:
        status = "contradicted"
        confidence = 0.8
    elif has_support:
        status = "supported"
        confidence = 0.75
    elif not tavily_answer.strip() and not tavily_results:
        status = "unknown"
        confidence = 0.0
    else:
        # Answer present but no clear cue words — treat as mixed/uncertain.
        status = "mixed"
        confidence = 0.4

    # Best Tavily result by score.
    best = max(tavily_results, key=lambda r: r.get("score", 0), default=None)

    return {
        "claim": claim,
        "status": status,
        "confidence": confidence,
        "source_url": best["url"] if best else None,
        "source_title": best["title"] if best else None,
        "summary": tavily_answer[:300] if tavily_answer else "No answer available.",
    }


def parse_verdict_response(
    raw: str,
    claim: str,
    tavily_answer: str,
    tavily_results: list[dict],
) -> dict:
    """Parse an LLM verdict reply into the canonical verdict dict.

    The LLM is instructed to reply as compact JSON:
        {"status": "contradicted"|"supported"|"mixed"|"unknown", "confidence": 0.0-1.0}

    Robustly handles:
    - Markdown code fences (```json ... ```)
    - Trailing prose after the JSON object
    - Invalid / unparseable replies  -> status="unknown", confidence=0.0
    - Missing "confidence" key       -> defaults to 0.0
    - Status value is lowercased and validated.
    """
    VALID_STATUSES = {"contradicted", "supported", "mixed", "unknown"}

    # Strip markdown code fences if present.
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    # Extract the first JSON object from the string (handles trailing prose).
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)

    parsed = None
    if match:
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            pass

    if parsed is None:
        status = "unknown"
        confidence = 0.0
    else:
        raw_status = str(parsed.get("status", "unknown")).lower().strip()
        status = raw_status if raw_status in VALID_STATUSES else "unknown"
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

    # Best Tavily result by score.
    best = max(tavily_results, key=lambda r: r.get("score", 0), default=None)

    return {
        "claim": claim,
        "status": status,
        "confidence": confidence,
        "source_url": best["url"] if best else None,
        "source_title": best["title"] if best else None,
        "summary": tavily_answer[:300] if tavily_answer else "No answer available.",
    }


# ---------------------------------------------------------------------------
# API adapters
# ---------------------------------------------------------------------------

def judge_verdict(claim: str, tavily_answer: str, tavily_results: list[dict]) -> dict:
    """Call Nebius LLM to judge the verdict and return the canonical verdict dict."""
    client = _get_nebius_client()

    system_prompt = (
        'You are a fact-checker. Given a CLAIM and web SEARCH RESULTS, decide if the '
        'search results CONTRADICT, SUPPORT, or are MIXED/inconclusive about the claim. '
        'Reply with ONLY compact JSON: {"status": "contradicted"|"supported"|"mixed"|"unknown", '
        '"confidence": <0.0-1.0>}. '
        'Judge the FULL assertion as a true/false statement — not whether its words '
        'merely appear in the results. The words of a claim co-occurring in a result '
        '(e.g. a travel page mentioning both "Paris" and "Belgium") is NOT support. '
        'Use "contradicted" ONLY when the evidence DIRECTLY shows the specific '
        'assertion is FALSE. '
        'Use "supported" ONLY when the evidence directly affirms the claim is TRUE. '
        'Use "mixed" when evidence is conflicting. '
        'Use "unknown" when the results are neutral, off-topic, or do not actually '
        'address whether the assertion is true. '
        'CRITICAL — absence of evidence is NOT contradiction. If the claim is VAGUE, '
        'subjective, or context-dependent, it is UNVERIFIABLE: return "unknown", NOT "contradicted". '
        'Unverifiable cases include: '
        '(1) Unresolved first-person or deictic subject — "we", "our", "this country", '
        '"our nation", "this company", "my team" with no named referent. Without knowing '
        'WHO "we/our" refers to, no search result can confirm or deny the assertion. '
        'Return "unknown" even if the rest of the claim sounds falsifiable. '
        '(2) Missing specific who/where/when — e.g. "we planted millions of trees across '
        'the nation" with no named nation, or "we funded schools". '
        'A claim only warrants "contradicted" when you can point to evidence that the '
        'SPECIFIC, IDENTIFIED assertion is plainly false. When in doubt, prefer "unknown".'
    )

    # Build user message with claim + top 2 result snippets for more context.
    top_snippets = "\n".join(
        f"[{i+1}] {r.get('content', '')[:400]}"
        for i, r in enumerate(tavily_results[:2])
    )
    # TODO(security): claim and tavily_answer are untrusted — consider length-capping and
    # stripping injection patterns before interpolating into the LLM user message.
    user_message = (
        f"CLAIM: {claim}\n\n"
        f"SEARCH ANSWER: {tavily_answer}\n\n"
        f"TOP RESULTS:\n{top_snippets}"
    )

    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=60,
    )

    raw = resp.choices[0].message.content or ""
    return parse_verdict_response(raw, claim, tavily_answer, tavily_results)


def extract_claim(transcript: str) -> str | None:
    """Call Nebius to extract the single most checkable claim from transcript."""
    client = _get_nebius_client()

    system_prompt = (
        "You extract claims for a fact-checker. Given a spoken turn, restate the "
        "SINGLE most checkable factual claim the speaker ASSERTED. "
        "CRITICAL: restate it FAITHFULLY, exactly as the speaker meant it, even if it "
        "is FALSE. Do NOT correct it, fix it, negate it, or add 'not'. If the speaker "
        "says 'Paris is in Belgium', output 'Paris is in Belgium' — never 'Paris is in "
        "France'. Your job is to capture what they claimed, not what is true. "
        "Reply with ONLY that one claim as a plain sentence — no preamble, no explanation. "
        "A claim is ONLY checkable when its subject is a specific, named entity "
        "(a country, person, company, etc.) that a search engine could look up. "
        "If the subject is an unresolved first-person or deictic reference — 'we', 'our', "
        "'our nation', 'our country', 'this country', 'my team', 'this company' — and "
        "no named referent is given in the turn, the claim is NOT checkable: reply NONE. "
        "If the turn contains no checkable factual claim, reply with exactly: NONE"
    )

    # TODO(security): transcript is raw speaker input — prompt injection possible if adversarial use.
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        temperature=0.2,
        max_tokens=120,
    )

    raw = resp.choices[0].message.content or ""
    return parse_claim_response(raw)


def fact_check(claim: str, timings: dict | None = None) -> dict:
    """Search Tavily for the claim and return a verdict via LLM judge.

    Uses judge_verdict (Nebius LLM) as the primary verdict engine.
    Falls back to score_verdict (heuristic) if judge_verdict raises any error.
    If `timings` is passed, records `tavily` and `judge` stage durations (s).
    """
    tv = _get_tavily_client()
    # Phrase as a verification question so Tavily's answer engine evaluates the
    # claim's truth, instead of returning pages where the claim's words merely
    # co-occur (which made false claims like "Paris is in Belgium" look supported).
    query = f"Is it true that {claim} Fact check this statement."
    t0 = time.perf_counter()
    r = tv.search(query=query, search_depth="fast", include_answer=True, max_results=5)
    if timings is not None:
        timings["tavily"] = time.perf_counter() - t0
    answer = r.get("answer", "")
    results = r.get("results", [])

    t1 = time.perf_counter()
    try:
        verdict = judge_verdict(claim, answer, results)
    except Exception as exc:
        print(f"[WARN] judge_verdict failed ({exc}), falling back to score_verdict", file=sys.stderr)
        verdict = score_verdict(claim, answer, results)
    if timings is not None:
        timings["judge"] = time.perf_counter() - t1
    return verdict


def check_turn(transcript: str, timings: dict | None = None) -> dict | None:
    """Orchestrate: extract claim -> fact-check -> return verdict or None.

    If `timings` is passed, records per-stage wall-clock durations (seconds)
    into it under keys: extract_claim, tavily, judge.
    """
    t0 = time.perf_counter()
    claim = extract_claim(transcript)
    if timings is not None:
        timings["extract_claim"] = time.perf_counter() - t0
    if claim is None:
        return None
    return fact_check(claim, timings)


# ---------------------------------------------------------------------------
# Rebuttal generation + TTS playback
# ---------------------------------------------------------------------------

def _fallback_rebuttal(summary: str) -> str:
    """Build a safe fallback rebuttal from the first sentence of summary.

    Pure helper — unit-testable without any network calls.
    """
    if not summary or summary.strip() == "No answer available.":
        return "Actually, that's not right."
    # Take only the first sentence.
    first_sentence = summary.split(".")[0].strip()
    if not first_sentence:
        return "Actually, that's not right."
    return "Actually, that's not right. " + first_sentence + "."


def generate_rebuttal(verdict: dict, barker_text: str | None = None) -> str:
    """Call Nebius to produce a short, punchy spoken rebuttal for a contradicted claim.

    `barker_text` is the full text of the barker that just played. The rebuttal
    must sound like a direct continuation of it — same speaker, same thought.
    Returns a 1-2 sentence plain-text string suitable for TTS.
    Falls back to _fallback_rebuttal if Nebius fails.
    """
    claim = verdict.get("claim", "")
    summary = verdict.get("summary", "")

    try:
        client = _get_nebius_client()

        if barker_text:
            system_prompt = (
                "You are a confident, witty heckler who just interrupted a speaker. "
                "You already said the opener (provided below) — the listener just heard it. "
                "Now deliver the actual correction: 1-2 punchy spoken sentences that flow "
                "naturally from where the opener left off, as if it is one continuous speech. "
                "Do NOT re-introduce yourself. Do NOT repeat the opener. Do NOT say 'Well actually' "
                "or any similar transition — just continue the thought and land the correction. "
                "No markdown. No URLs. No Citations. Plain spoken text only."
            )
            # TODO(security): claim and summary are LLM-derived from untrusted speaker input — sanitize
            # before production adversarial use.
            user_message = (
                f"OPENER YOU ALREADY SAID: {barker_text}\n\n"
                f"CLAIM (what they said): {claim}\n\n"
                f"FACTS (the correction): {summary}"
            )
        else:
            system_prompt = (
                "You are a confident, witty heckler catching someone in a factual falsehood. "
                "Write exactly 1-2 punchy spoken sentences correcting the claim using the provided facts. "
                "No preamble. No 'Well actually'. No markdown. No URLs. No citations. "
                "Just the correction with attitude — plain text only, as if you are speaking aloud."
            )
            user_message = (
                f"CLAIM (what they said): {claim}\n\n"
                f"FACTS (the correction): {summary}"
            )

        resp = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=80,
        )

        raw = resp.choices[0].message.content or ""
        return raw.strip()

    except Exception as exc:
        print(f"[WARN] generate_rebuttal failed ({exc}), using fallback", file=sys.stderr)
        return _fallback_rebuttal(summary)


async def speak(text: str, voice_id: str = "POBHtemksfWQbng0") -> bytes:
    """Call Gradium TTS and return WAV bytes.

    Async because client.tts() is a coroutine.
    """
    import gradium

    client = gradium.client.GradiumClient(api_key=os.environ["GRADIUM_API_KEY"])
    result = await client.tts(
        setup={"voice_id": voice_id, "output_format": "wav"},
        text=text,
    )
    return result.raw_data


def play_audio(wav_bytes: bytes) -> None:
    """Play WAV bytes through the default output device via sounddevice.

    Reads sample rate and channel count from the RIFF header, then reads PCM
    payload starting at byte 44 (canonical WAV header size) to work around
    Gradium's unreliable RIFF size field.
    """
    import sounddevice as sd

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()

    pcm = wav_bytes[WAV_HEADER_BYTES:]

    with sd.RawOutputStream(samplerate=sr, channels=ch, dtype="int16") as out:
        out.write(pcm)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    _REQUIRED_ENV = ("GRADIUM_API_KEY", "NEBIUS_API_KEY", "TAVILY_API_KEY")
    _missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if _missing:
        sys.exit(f"Missing required environment variables: {', '.join(_missing)}\nCopy .env.example to .env and fill in your keys.")

    if len(sys.argv) > 1:
        transcript = " ".join(sys.argv[1:])
    else:
        transcript = "The Great Wall of China is visible from space with the naked eye."

    print(f"[INPUT] {transcript}\n")
    verdict = check_turn(transcript)

    if verdict is None:
        print("[RESULT] No checkable claim found in this turn.")
    else:
        print(json.dumps(verdict, indent=2))
