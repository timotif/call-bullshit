"""Generate barker WAV files for the Call Bullshit voice agent."""
import asyncio
import json
import os
from pathlib import Path

import gradium
from dotenv import load_dotenv

load_dotenv()

# Each entry: (voice_id, voice_label, text)
# Voices picked for interrupt energy across US + UK accents.
# Texts graded by length so the runtime can pick the shortest that covers
# the session's measured rebuttal-prep latency. With streaming TTS the budget
# is now ~1-2s (was ~4-17s), so the short tier below is what normally fires;
# the longer tiers remain as cover for slow/cold rounds where the budget spikes.
BARKERS = [
    # ── ~1-2s (streaming-era default: quick interjection, rebuttal lands fast) ─
    ("POBHtemksfWQbng0", "garrett",
     "Whoa — no. That's wrong."),

    ("6MFfc37kq0sBjBjy", "sterling",
     "Nope. Not true."),

    ("dME3IWyZBvmh1n1q", "toby",
     "Hold on — that's wrong."),

    ("6PWnV0Nq4wu7RVBT", "maeve",
     "Wait — that's not right."),

    ("_6Aslh2DxfmnRLmP", "russell",
     "Hang on — no."),

    ("CF0NgaMwHMMrHZn0", "reuben",
     "Stop — that's false."),

    # ── ~4s ─────────────────────────────────────────────────────────────────
    ("POBHtemksfWQbng0", "garrett",
     "Whoa — hold on. That's not right. Let me stop you there."),

    ("6MFfc37kq0sBjBjy", "sterling",
     "Nope, nope — stop right there. That one's wrong."),

    ("dME3IWyZBvmh1n1q", "toby",
     "Hold on — I'm going to have to stop you there. That's not accurate."),

    ("6PWnV0Nq4wu7RVBT", "maeve",
     "Excuse me — that's actually not correct. Let me jump in here."),

    # ── ~6s ─────────────────────────────────────────────────────────────────
    ("_6Aslh2DxfmnRLmP", "russell",
     "Wait, wait — that doesn't check out. Give me one second, "
     "because I have to correct that right now."),

    ("CF0NgaMwHMMrHZn0", "reuben",
     "Hang on — I can't let that slide. What you just said is not accurate, "
     "and I need to set the record straight."),

    ("uem82D50GRv2Dwma", "pippa",
     "Actually — I'm sorry — that's not right. I've looked into this and "
     "the facts tell a very different story."),

    ("KUpE0JVhjiIzp1Fk", "damon",
     "Hold up — that's just not true. I was sitting here quietly but "
     "I can't let that one go. Here's what actually happened."),

    # ── ~8s ─────────────────────────────────────────────────────────────────
    ("r2sIQdqqoqgRJuXw", "marcus",
     "Hey, hey — stop right there. I cannot let that slide, because what you "
     "just said is simply not true. I've done my homework on this. Here's the deal."),

    ("POBHtemksfWQbng0", "garrett",
     "Okay — time out. I've been listening patiently and that claim is wrong. "
     "I actually looked this up, so let me give you the real picture here."),

    ("dME3IWyZBvmh1n1q", "toby",
     "Sorry to interrupt — but I genuinely cannot let that stand. That's not what "
     "the evidence shows, not even close. Let me correct the record quickly."),

    ("4SZHfMpw-p46Ywgs", "harper",
     "Whoa, whoa — pause for a second. That's not accurate, and if we just let it "
     "slide, it becomes the version people remember. So let me clear this up."),

    # ── ~10s ────────────────────────────────────────────────────────────────
    ("6MFfc37kq0sBjBjy", "sterling",
     "Okay — no, no, no. Time out. Everybody take a breath. Because what was just "
     "said does not check out — not even a little. I've actually looked into this "
     "and the real story is pretty different. Let me set you straight on this one."),

    ("_6Aslh2DxfmnRLmP", "russell",
     "Alright — I have to stop you right there, because that claim is flat-out wrong "
     "and I can back that up. I know this sounds blunt, but if I don't say something "
     "now someone in this room is going to walk out believing something false. Here's the truth."),

    ("CF0NgaMwHMMrHZn0", "reuben",
     "Right — I'm going to have to jump in here, because that's one of those statements "
     "that sounds plausible but really doesn't hold up once you look at the actual data. "
     "I'm not trying to be difficult — I just think it matters that we get this right. So."),

    ("6PWnV0Nq4wu7RVBT", "maeve",
     "Oh — I'm so sorry to cut you off, but I genuinely cannot sit here and let that "
     "go unchallenged. That's actually been looked into quite carefully, and the findings "
     "are very different from what you're suggesting. Let me walk you through it."),

    ("uem82D50GRv2Dwma", "pippa",
     "Okay stop — I hate to interrupt, but that claim really needs a correction, and the "
     "sooner the better. It's one of those things that circulates as a fact even though "
     "it's been thoroughly checked and it just doesn't hold up. Here's the actual situation."),

    ("KUpE0JVhjiIzp1Fk", "damon",
     "Whoa — hold on a second. I've been patient, but that one crossed a line for me. "
     "That claim has been out there for a while and it still isn't true. I actually looked "
     "this up before coming here today because I suspected it might come up. So here we go."),

    # ── ~13s ────────────────────────────────────────────────────────────────
    ("r2sIQdqqoqgRJuXw", "marcus",
     "Whoa, whoa, whoa — stop the presses, back it up. Did you really just say that out loud "
     "like it was established fact? Oh no. We are not doing that today. I have been sitting "
     "here very patiently and I just cannot anymore. That is not how this went. "
     "Here is exactly what the evidence actually shows, and I promise it's worth the thirty seconds."),

    ("dME3IWyZBvmh1n1q", "toby",
     "Hang on a moment — I really do have to push back on that, and not just a little, a lot. "
     "Because if we let that stand, somebody is going to repeat it at a dinner party next week "
     "and it's going to spread. I've seen it happen. So for the record, and I'll be quick about "
     "this, let me walk you through what's actually real here."),

    ("4SZHfMpw-p46Ywgs", "harper",
     "Okay — I need everyone to pause for a moment, because what was just said is the kind of "
     "claim that sounds very confident and very specific but actually doesn't survive even basic "
     "scrutiny. I don't say this to be unkind. I say it because accuracy genuinely matters here, "
     "especially when people are making decisions based on what they hear. So here's the real picture."),

    # ── ~17s ────────────────────────────────────────────────────────────────
    ("6MFfc37kq0sBjBjy", "sterling",
     "Alright — everybody stop what you're doing for just a moment, because I have to address what "
     "was just said, and I want to do it properly. That claim has been floating around for a long "
     "time — I've heard it in meetings, I've seen it in presentations, I've watched it get cited "
     "like gospel — and the problem is it's simply not supported by the evidence. I know that's "
     "uncomfortable. I know it sounds like I'm being difficult. But the actual facts here are "
     "genuinely fascinating and they tell a completely different story. So please, bear with me "
     "for about thirty seconds, because this is worth getting right. Here's what we actually know."),
]

OUT_DIR = Path(__file__).parent / "barkers"


async def generate_one(client, idx: int, voice_id: str, voice_label: str, text: str) -> dict:
    filename = f"barker_{idx:02d}_{voice_label}.wav"
    path = os.path.join(OUT_DIR, filename)
    print(f"  [{idx:02d}] {voice_label}: {text[:60]}…")
    result = await client.tts(
        setup={"voice_id": voice_id, "output_format": "wav"},
        text=text,
    )
    with open(path, "wb") as f:
        f.write(result.raw_data)
    print(f"       -> {filename}  ({len(result.raw_data):,} bytes)")
    return {"file": filename, "text": text, "voice": voice_label, "voice_id": voice_id}


async def main():
    client = gradium.client.GradiumClient(api_key=os.environ["GRADIUM_API_KEY"])
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Generating {len(BARKERS)} barkers across {len({v for _,v,_ in BARKERS})} voices…\n")

    # Generate sequentially to avoid hammering the TTS API
    manifest = []
    for idx, (voice_id, voice_label, text) in enumerate(BARKERS):
        entry = await generate_one(client, idx, voice_id, voice_label, text)
        manifest.append(entry)

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nDone. {len(manifest)} barkers written to {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
