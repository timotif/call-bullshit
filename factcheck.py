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
_gradium_client: "GradiumClient | None" = None

# Model selection, all overridable via env so slugs aren't hardcoded.
# All three legs run on Qwen3-30B-A3B-Instruct-2507 (MoE, ~3B active params).
# Tavily supplies the evidence; the judge mostly classifies it. Benchmarked vs
# Llama-3.3-70B: matched judge accuracy on the hard cases (incl. the deictic-
# subject rule) while cutting judge ~1250ms->~380ms and extraction ~870ms->~300ms.
# For rebuttal-gen it produced finished, wittier output in ~730ms vs the 70B's
# ~4s. NOTE: use the *-Instruct* (non-thinking) Qwen, NOT a *-Thinking* or
# nemotron/nemotron-style reasoning model — those spend the whole token budget
# reasoning and return EMPTY content under our short max_tokens, i.e. dead air.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
REBUTTAL_MODEL = os.environ.get("REBUTTAL_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

# Tavily fetch + how much of each result the judge sees. Latency is flat across
# 3-5 results (measured), so we fetch 5 and feed the judge ALL of them — fetched
# evidence shouldn't be discarded. SNIPPET_CHARS bounds each result's prompt cost.
TAVILY_MAX_RESULTS = int(os.environ.get("TAVILY_MAX_RESULTS", "5"))
SNIPPET_CHARS = int(os.environ.get("SNIPPET_CHARS", "400"))


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


def _get_gradium_client() -> "GradiumClient":
    """Lazily build and cache one GradiumClient (TTS).

    speak()/speak_stream() previously built a fresh client on EVERY call, paying
    connection/handshake setup per rebuttal — a per-call latency leak on the hot
    path. Cache it like the Nebius/Tavily clients so it's created once.
    """
    global _gradium_client
    if _gradium_client is None:
        import gradium
        _gradium_client = gradium.client.GradiumClient(api_key=os.environ["GRADIUM_API_KEY"])
    return _gradium_client


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
        "source_url": best.get("url") if best else None,
        "source_title": best.get("title") if best else None,
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
        "source_url": best.get("url") if best else None,
        "source_title": best.get("title") if best else None,
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

    # Feed the judge every result Tavily returned, not just the top 2 — if Tavily
    # spent the latency fetching them (measured identical for 3 vs 5 results), the
    # judge should weigh them, especially when sources conflict. Each snippet is
    # capped at SNIPPET_CHARS so the prompt stays bounded regardless of result count.
    snippets = "\n".join(
        f"[{i+1}] {r.get('content', '')[:SNIPPET_CHARS]}"
        for i, r in enumerate(tavily_results)
    )
    # TODO(security): claim and tavily_answer are untrusted — consider length-capping and
    # stripping injection patterns before interpolating into the LLM user message.
    user_message = (
        f"CLAIM: {claim}\n\n"
        f"SEARCH ANSWER: {tavily_answer}\n\n"
        f"TOP RESULTS:\n{snippets}"
    )

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
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
        model=EXTRACT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        temperature=0.2,
        max_tokens=60,
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
    r = tv.search(query=query, search_depth="fast", include_answer=True, max_results=TAVILY_MAX_RESULTS)
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


def warm_up_verdict_path() -> None:
    """Fire one throwaway extract + judge to spin up the Nebius models at startup.

    Qwen3-30B-A3B (extract + judge model) pays a cold-start penalty on its first
    request after idle (~2s vs ~0.9s warm). Calling this at startup moves that
    penalty off the first REAL claim so the demo's first 'caught you' is fast.
    No Tavily call — we feed canned evidence, since only the LLM legs cold-start.
    Best-effort: swallows errors so a warm-up hiccup never blocks startup.
    """
    try:
        extract_claim("The Eiffel Tower is located in Paris, France.")
        judge_verdict(
            "The Eiffel Tower is in Berlin.",
            "The Eiffel Tower is located in Paris, France, not Berlin.",
            [{
                "content": "The Eiffel Tower is a landmark in Paris, France.",
                "score": 0.9,
                "url": "https://example.com/eiffel",
                "title": "Eiffel Tower",
            }],
        )
    except Exception as exc:
        print(f"[WARN] verdict-path warm-up failed ({exc})", file=sys.stderr)


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
    Returns ONE short, witty, fact-first sentence suitable for TTS — now that
    interruptions fire fast, the barker is a quick interjection and the rebuttal
    goes straight to the correction.
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
                "Now land the correction in ONE short spoken sentence that flows naturally "
                "from where the opener left off, as if it is one continuous speech. Lead with "
                "the fact — state what's actually true, with a touch of attitude. "
                "Do NOT re-introduce yourself. Do NOT repeat the opener. Do NOT say 'Well actually' "
                "or any similar transition. Keep it tight — one sentence, no rambling. "
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
                "Write exactly ONE short, punchy spoken sentence that corrects the claim, leading "
                "with the actual fact. A touch of attitude, but tight and fact-first — no rambling. "
                "No preamble. No 'Well actually'. No markdown. No URLs. No citations. "
                "Just the correction, plain text only, as if you are speaking aloud."
            )
            user_message = (
                f"CLAIM (what they said): {claim}\n\n"
                f"FACTS (the correction): {summary}"
            )

        resp = client.chat.completions.create(
            model=REBUTTAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=50,  # one fact-first sentence — keep it tight
        )

        raw = resp.choices[0].message.content or ""
        return raw.strip()

    except Exception as exc:
        print(f"[WARN] generate_rebuttal failed ({exc}), using fallback", file=sys.stderr)
        return _fallback_rebuttal(summary)


async def speak(text: str, voice_id: str = "POBHtemksfWQbng0") -> bytes:
    """Call Gradium TTS (buffered) and return WAV bytes.

    Async because client.tts() is a coroutine. Retained for back-compat / any
    non-streaming caller; the live rebuttal path now uses speak_stream() so it
    can start playing on the first chunk instead of after the whole WAV. Uses
    the cached client (no per-call connection churn).
    """
    client = _get_gradium_client()
    result = await client.tts(
        setup={"voice_id": voice_id, "output_format": "wav"},
        text=text,
    )
    return result.raw_data


async def open_tts_stream(text: str, voice_id: str = "POBHtemksfWQbng0"):
    """Open a streaming Gradium TTS request and return the raw TTSStream.

    The TTSStream exposes `.sample_rate` (the device rate to open playback at —
    the caller MUST use this, NOT a hardcoded rate: the live endpoint has been
    observed at 48 kHz; fall back to main.SAMPLE_RATE only if it is unset) and
    `.iter_bytes()` (async iterator of raw 16-bit signed LE PCM chunks, mono,
    at `.sample_rate` — already base64-DECODED by the SDK).
    We request output_format="pcm", NOT "wav", so there is no per-chunk RIFF
    header to strip mid-stream (only wav's first chunk carries the 44-byte
    header — fragile). The point of streaming is time-to-first-chunk (~606ms
    warm) vs the buffered tts() (~3.7s): playback can begin on chunk one.

    Use this when you need the sample rate (e.g. to open the output device);
    use speak_stream() when you only need the chunk iterator. Uses the cached
    client (no per-call connection churn).
    """
    client = _get_gradium_client()
    return await client.tts_stream(
        setup={"voice_id": voice_id, "output_format": "pcm"},
        text=text,
    )


async def speak_stream(text: str, voice_id: str = "POBHtemksfWQbng0"):
    """Stream Gradium TTS, yielding raw PCM chunks as synthesis produces them.

    Contract: this is an ASYNC GENERATOR. The caller does
    `async for chunk in speak_stream(...)` and each chunk is raw 16-bit signed
    LE PCM, mono, at the stream's `.sample_rate` (the live endpoint has been
    observed at 48 kHz; do NOT assume 24 kHz). Thin wrapper over open_tts_stream
    for callers that don't need the sample rate.

    NOTE: main.py's live rebuttal path uses open_tts_stream directly (it needs
    the sample rate to open the output device); this wrapper is kept as a
    convenience for external callers that only want the chunk iterator.

    Errors from Gradium propagate out of the generator (e.g. a stream that dies
    mid-synthesis after some audio has played); the playback drainer closes the
    device and the turn caller resets `speaking`.
    """
    stream = await open_tts_stream(text, voice_id=voice_id)
    async for chunk in stream.iter_bytes():
        yield chunk


def play_audio(wav_bytes: bytes) -> None:
    """Play WAV bytes through the default output device via sounddevice.

    Reads sample rate and channel count from the RIFF header, then reads PCM
    payload starting at byte 44 (canonical WAV header size) to work around
    Gradium's unreliable RIFF size field.

    Retained for the buffered/WAV path (barkers still load on-disk WAVs).
    The streaming rebuttal path uses play_audio_stream instead.
    """
    import sounddevice as sd

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()

    pcm = wav_bytes[WAV_HEADER_BYTES:]

    with sd.RawOutputStream(samplerate=sr, channels=ch, dtype="int16") as out:
        out.write(pcm)


def play_audio_stream(chunk_iter, sample_rate: int, channels: int = 1, prebuffer_chunks: int = 0) -> None:
    """Drain raw PCM chunks to one output stream, writing each as it arrives.

    `chunk_iter` is any (sync) iterable yielding raw 16-bit signed LE PCM bytes
    — the rebuttal path requests output_format="pcm" precisely so there is NO
    per-chunk RIFF header to strip (unlike the buffered WAV path). Each
    RawOutputStream.write() BLOCKS until the device accepts the bytes, so it
    paces playback to real time while later chunks are still being synthesized
    upstream; that blocking is also why the caller runs this in an executor.

    `prebuffer_chunks` holds the first N chunks before the first write, so a
    slow/cold synthesis round can't underrun the device (audible gap/glitch)
    the instant playback outpaces delivery. Once primed, writes resume one
    chunk at a time. In the live sequencing the barker plays during synthesis
    and is itself a natural prebuffer, so a small N (0-2) is plenty here; the
    knob exists so the cold/slow round — not just the warm one — is covered.
    The prebuffer only shifts WHEN the first write happens; it never drops,
    reorders, or duplicates chunks, and a stream shorter than N still flushes
    fully.

    Resource safety: the `with` opens exactly one stream for the whole rebuttal
    and closes it on the way out — on normal completion, on a write() device
    error, AND when `chunk_iter` itself raises mid-stream (a TTS stream can die
    after some audio has played). We deliberately let the exception propagate so
    the turn's caller can reset `speaking` and recover, but the device is never
    leaked. An empty iterator simply opens-and-closes the stream (no audio).
    """
    import sounddevice as sd

    it = iter(chunk_iter)
    with sd.RawOutputStream(samplerate=sample_rate, channels=channels, dtype="int16") as out:
        # Prime the prebuffer: pull up to N chunks before the first write so a
        # slow first round doesn't underrun. A short stream just primes fully.
        primed: list[bytes] = []
        for chunk in it:
            primed.append(chunk)
            if len(primed) >= prebuffer_chunks:
                break
        for chunk in primed:
            out.write(chunk)
        # Stream the remainder one chunk at a time; write() paces to real time.
        for chunk in it:
            out.write(chunk)


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
