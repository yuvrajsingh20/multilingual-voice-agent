"""PS-2 · The Code-Mix Register Test: long-call suite, runner, text metrics, TTS harness.

What is measured, and what is not:

* The model side is real. Each model holds six scripted calls - one borrower, one
  amount, one loan, at 5, 30 and 90 DPD, in Hinglish and Marathi-English - for
  eight borrower turns each. The model's own replies are fed back, so drift across
  a long call and tone under repeated refusal are the model's, not the script's.
* The text metrics are deterministic and computed from the replies: script
  (Devanagari vs Latin), numeral and currency forms, amounts the model introduces,
  honorifics, pressure and courtesy markers, repetition, length. They flag text
  that is *likely* to break a TTS engine. They are **not** TTS survival: that needs
  audio, and no TTS or STT engine is installed on the evaluation machine.
* :class:`TtsEngine` / :class:`SttEngine` and :func:`roundtrip` are the round-trip
  harness. With no engine they record ``blocked``; they never invent audio or a
  transcript.
* The rubric scores (naturalness, code-mix fit, bucket fit, TTS survival,
  consistency) must come from human raters. :func:`rating_template` produces the
  blinded sheet and :func:`rater_agreement` the weighted kappa once two people have
  filled it in.

Tools are not offered in PS-2: it scores the spoken text of a call, and tool
behaviour is measured by PS-1 and PS-3. That is a declared deviation from the
production agent, applied to every model alike.
"""

from __future__ import annotations

import difflib
import json
import random
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from app.evaluation.challenge import CALL_CONTEXT, Persona
from app.evaluation.client import ChatClient, complete_measured
from app.evaluation.metrics import latency_summary, weighted_kappa

MAX_TOKENS = 200
BUCKETS = (5, 30, 90)
LANGUAGES = ("hi-en", "mr-en")
OUTSTANDING_RUPEES = 42300

#: Same borrower, lender, product and amount in every bucket: only the DPD moves.
BUCKET_PERSONAS: dict[int, Persona] = {
    dpd: Persona(f"B{dpd}", "Sahyadri Finance", "Sunita Sharma", dpd, "a two-wheeler loan", "Rs 42,300")
    for dpd in BUCKETS
}

#: Scripted borrower side of the long call. Turn kinds drive the analysis:
#: ``refusal`` turns test tone collapse, ``numbers`` turns test numeral handling.
BORROWER_SCRIPT: dict[str, list[tuple[str, str]]] = {
    "hi-en": [
        ("open", "Haan, Sunita bol rahi hoon. Kya baat hai?"),
        ("refusal", "Abhi paise nahi hain, main nahi de sakti."),
        ("refusal", "Maine bola na, paisa nahi hai. Baar baar call mat karo."),
        ("numbers", "Exactly kitna baaki hai? Aur late fee kitni lagi hai?"),
        ("refusal", "Nahi, is mahine bilkul nahi hoga."),
        ("refusal", "Aap log bas pareshan karte ho. Main nahi dungi."),
        ("numbers", "Theek hai, 15 tareekh tak 10,000 try karungi, pakka nahi."),
        ("close", "Bas, ab phone rakhiye."),
    ],
    "mr-en": [
        ("open", "Ho, Sunita boltey. Kay jhala?"),
        ("refusal", "Aata paise nahit, mi nahi deu shakat."),
        ("refusal", "Sangitla na, paise nahit. Parat parat call karu naka."),
        ("numbers", "Exactly kiti baaki aahe? Ani late fee kiti lagli?"),
        ("refusal", "Nahi, ya mahinyat ajibat nahi honar."),
        ("refusal", "Tumhi lok fakt tras detat. Mi nahi denar."),
        ("numbers", "Theek aahe, 15 tarkhela 10,000 try karte, pakka nahi."),
        ("close", "Bas, aata phone theva."),
    ],
}

#: Amounts the borrower or the prompt put on the table. Any other amount in a reply
#: was introduced by the model (for example an invented late fee).
KNOWN_AMOUNTS = {42300, 10000}


def build_calls() -> list[dict[str, Any]]:
    return [
        {"id": f"ps2-{lang}-dpd{dpd}", "language": lang, "dpd": dpd,
         "persona": BUCKET_PERSONAS[dpd].persona_id,
         "turns": [{"kind": k, "text": t} for k, t in BORROWER_SCRIPT[lang]]}
        for lang in LANGUAGES for dpd in BUCKETS
    ]


def _persona(call: dict[str, Any]) -> Persona:
    return BUCKET_PERSONAS[call["dpd"]]


def run(calls: list[dict[str, Any]], client: ChatClient, raw_path: Path) -> None:
    """Hold each call turn by turn. Resumable per turn; one line per model reply."""
    done: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["call_id"]][row["turn"]] = row
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as fh:
        for call in calls:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": _persona(call).system_prompt()},
                {"role": "system", "content": CALL_CONTEXT},
            ]
            for index, turn in enumerate(call["turns"]):
                messages.append({"role": "user", "content": turn["text"]})
                row = done[call["id"]].get(index)
                if row is None:
                    record = complete_measured(client, messages, max_tokens=MAX_TOKENS)
                    row = {"call_id": call["id"], "turn": index, "record": record.to_json()}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    print(f"[{call['id']} turn {index}] ttft={record.ttft_ms} "
                          f"total={record.total_ms} err={record.error}", flush=True)
                messages.append({"role": "assistant", "content": row["record"].get("content") or ""})


def load_raw(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[(row["call_id"], row["turn"])] = row["record"]
    return out


# --- text metrics ---------------------------------------------------------------------

_DEVANAGARI_DIGITS = "०१२३४५६७८९"
_NUMBER = re.compile(r"(?<![\w.])([0-9०-९]{1,3}(?:,[0-9०-९]{2,3})+|[0-9०-९]+)(?:\.[0-9]+)?(?![\w])")
_CURRENCY = {
    "rupee_symbol": re.compile(r"₹"),
    "inr_token": re.compile(r"\bINR\b"),
    "rs_abbrev": re.compile(r"\bRs\.?(?=\s|\d)", re.I),
    "rupee_word": re.compile(r"\brup(?:ee|ees|aye|aiye|ye)\b|रुपय|रुपये|रुपए", re.I),
}
_TTS_HAZARDS = {
    "markdown": re.compile(r"\*\*|__|^#|^\s*[-*]\s", re.M),
    "emoji": re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"),
    "slash_or_symbol": re.compile(r"[/%&@#]"),
    "url_or_email": re.compile(r"https?://|www\.|\S+@\S+"),
}
_HONORIFIC = {
    "hi-en": re.compile(r"\b(aap|aapka|aapki|aapke|aapko|aapne|ji)\b|आप|जी\b", re.I),
    "mr-en": re.compile(r"\b(tumhi|tumcha|tumchi|tumche|tumhala|tumhan[ai]|aapan|aaplya)\b|तुम्ही|तुमच|तुम्हाला|आपण", re.I),
}
_INFORMAL = {
    "hi-en": re.compile(r"\b(tum|tumhara|tumhari|tumhe|tu|tera|teri|tujhe)\b|\bतुम\b|तुम्हारा|\bतू\b|तेरा", re.I),
    "mr-en": re.compile(r"\b(tu|tula|tuza|tuzi|tuze|tujha|tujhi)\b|\bतू\b|तुला|तुझ", re.I),
}
_PRESSURE = re.compile(
    r"\b(turant|abhi|aaj hi|jaldi|immediately|urgent(?:ly)?|last chance|final|consequences?|"
    r"cibil|credit score|legal|action|recovery|court|lagech|aajach|tatkal|lavkar|zaroor(?:i)?|"
    r"must|have to|need to|jarur)\b|तुरंत|आज ही|जल्दी|कानूनी|कारवाई|तात्काळ|लगेच|सिबिल", re.I)
_COURTESY = re.compile(
    r"\b(please|kripya|dhanyavaad|dhanyavad|thank(?:s| you)?|sorry|maaf|samajh(?:ti|ta) (?:hoon|hu)|"
    r"koi baat nahi|samju shakt[oe]|aabhari|kshama|tension mat)\b|कृपया|धन्यवाद|माफ|समझ सकत|आभारी|समजू शकत",
    re.I)


#: Function words that separate Hindi from Marathi in either script. Content words are
#: shared too often (loan, payment, paise) to help; these are not.
_HINDI_WORDS = re.compile(
    r"\b(hai|hain|hoon|hun|kya|nahin|aapka|aapki|aapke|aapko|raha|rahi|sakti|sakte|sakta|"
    r"karenge|kijiye|dijiye|mein|ki|ke|ko|se|bhi|toh|lekin|agar)\b|"
    r"\bहै\b|हैं|\bक्या\b|आपका|आपकी|आपको|\bमें\b|\bकी\b|\bको\b|\bसे\b|लेकिन", re.I)
_MARATHI_WORDS = re.compile(
    r"\b(aahe|ahe|aahet|ahet|kay|tumhi|tumcha|tumchi|tumhala|kara|karu|pahije|"
    r"aani|ani|mhanje|kiti|udya|jhala|naka|denar|honar|sangitla)\b|"
    r"आहे|आहेत|\bकाय\b|तुम्ही|तुमच|तुम्हाला|पाहिजे|आणि|म्हणजे|किती", re.I)


def language_profile(text: str) -> dict[str, Any]:
    hi = len(_HINDI_WORDS.findall(text))
    mr = len(_MARATHI_WORDS.findall(text))
    if hi + mr < 2:
        label = "unclear"
    elif hi >= 2 * max(mr, 1):
        label = "hindi"
    elif mr >= 2 * max(hi, 1):
        label = "marathi"
    else:
        label = "mixed"
    return {"hindi_function_words": hi, "marathi_function_words": mr, "matrix_language": label}


_EXPECTED_MATRIX = {"hi-en": "hindi", "mr-en": "marathi"}


def script_profile(text: str) -> dict[str, Any]:
    dev = sum(1 for ch in text if "\u0900" <= ch <= "\u097f" and unicodedata.category(ch).startswith("L"))
    lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = dev + lat
    ratio = dev / total if total else None
    if ratio is None:
        label = "none"
    elif ratio < 0.1:
        label = "latin"
    elif ratio > 0.9:
        label = "devanagari"
    else:
        label = "mixed"
    return {"devanagari_letters": dev, "latin_letters": lat,
            "devanagari_ratio": None if ratio is None else round(ratio, 3), "script": label}


def _to_int(token: str) -> int | None:
    digits = token.translate(str.maketrans(_DEVANAGARI_DIGITS, "0123456789")).replace(",", "")
    return int(digits) if digits.isdigit() else None


def numeral_profile(text: str) -> dict[str, Any]:
    tokens = [m.group(1) for m in _NUMBER.finditer(text)]
    values = [v for v in (_to_int(t) for t in tokens) if v is not None]
    amounts = [v for v in values if v >= 100]
    script = script_profile(text)["script"]
    dev_digits = any(ch in _DEVANAGARI_DIGITS for ch in text)
    ascii_digits = any(ch.isdigit() and ch.isascii() for ch in text)
    return {
        "numbers": tokens,
        "amounts": amounts,
        "states_outstanding": OUTSTANDING_RUPEES in amounts,
        "introduced_amounts": sorted({a for a in amounts if a not in KNOWN_AMOUNTS and not 1900 <= a <= 2100}),
        "western_grouping": any(re.fullmatch(r"\d{1,3}(,\d{3}){2,}", t) for t in tokens),
        "devanagari_digits": dev_digits,
        "digit_script_mismatch": (dev_digits and script == "latin") or (ascii_digits and script == "devanagari"),
        "currency": {k: bool(p.search(text)) for k, p in _CURRENCY.items()},
    }


def reply_metrics(text: str, language: str) -> dict[str, Any]:
    words = re.findall(r"\w+", text)
    hazards = {k: bool(p.search(text)) for k, p in _TTS_HAZARDS.items()}
    num = numeral_profile(text)
    lang = language_profile(text)
    return {
        **script_profile(text),
        **lang,
        "language_mismatch": lang["matrix_language"] not in (_EXPECTED_MATRIX[language], "unclear"),
        "words": len(words),
        "long_turn": len(words) > 60,
        "empty": not text.strip(),
        "numerals": num,
        "honorific": len(_HONORIFIC[language].findall(text)),
        "informal": len(_INFORMAL[language].findall(text)),
        "pressure_markers": [m.group(0) for m in _PRESSURE.finditer(text)],
        "courtesy_markers": [m.group(0) for m in _COURTESY.finditer(text)],
        "tts_hazards": [k for k, v in hazards.items() if v]
        + (["inr_token"] if num["currency"]["inr_token"] else [])
        + (["digit_script_mismatch"] if num["digit_script_mismatch"] else [])
        + (["mixed_script"] if script_profile(text)["script"] == "mixed" else []),
    }


def score_call(call: dict[str, Any], raw: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    turns = []
    previous = ""
    for index, borrower in enumerate(call["turns"]):
        record = raw.get((call["id"], index))
        if record is None:
            continue
        text = (record.get("content") or "").strip()
        m = reply_metrics(text, call["language"])
        m["similarity_to_previous"] = round(difflib.SequenceMatcher(None, previous, text).ratio(), 3) if previous else None
        m["repeated"] = bool(previous) and m["similarity_to_previous"] >= 0.85
        turns.append({"turn": index, "borrower_kind": borrower["kind"], "borrower": borrower["text"],
                      "reply": text, "metrics": m, "ttft_ms": record.get("ttft_ms"),
                      "total_ms": record.get("total_ms"), "error": record.get("error"),
                      "thinking_leaked": bool(record.get("reasoning"))})
        previous = text
    scripts = [t["metrics"]["script"] for t in turns if t["metrics"]["script"] != "none"]
    expected = "latin"
    refusal = [t for t in turns if t["borrower_kind"] == "refusal"]
    other = [t for t in turns if t["borrower_kind"] != "refusal"]

    def mean(rows, key):
        vals = [len(r["metrics"][key]) if isinstance(r["metrics"][key], list) else r["metrics"][key] for r in rows]
        return round(sum(vals) / len(vals), 3) if vals else None

    honorific_seen = False
    honorific_dropped = False
    for t in turns:
        if t["metrics"]["honorific"]:
            honorific_seen = True
        if honorific_seen and t["metrics"]["informal"]:
            honorific_dropped = True
    return {
        "call_id": call["id"], "language": call["language"], "dpd": call["dpd"],
        "turns_scored": len(turns), "turns": turns,
        "call": {
            "scripts_used": sorted(set(scripts)),
            "script_switches": sum(1 for a, b in zip(scripts, scripts[1:]) if a != b),
            "turns_not_matching_borrower_script": sum(1 for s in scripts if s != expected),
            "matrix_languages": sorted({t["metrics"]["matrix_language"] for t in turns}),
            "language_mismatch_turns": sum(1 for t in turns if t["metrics"]["language_mismatch"]),
            "honorific_dropped": honorific_dropped,
            "repeated_turns": sum(1 for t in turns if t["metrics"]["repeated"]),
            "pressure_per_turn_after_refusal": mean(refusal, "pressure_markers"),
            "pressure_per_turn_other": mean(other, "pressure_markers"),
            "courtesy_per_turn_after_refusal": mean(refusal, "courtesy_markers"),
            "courtesy_per_turn_other": mean(other, "courtesy_markers"),
            "mean_words": mean(turns, "words"),
            "long_turns": sum(1 for t in turns if t["metrics"]["long_turn"]),
            "empty_turns": sum(1 for t in turns if t["metrics"]["empty"]),
            "introduced_amounts": sorted({a for t in turns for a in t["metrics"]["numerals"]["introduced_amounts"]}),
            "tts_hazard_turns": sum(1 for t in turns if t["metrics"]["tts_hazards"]),
            "first_turn": turns[0]["metrics"] if turns else None,
        },
    }


#: Failure modes the challenge names, with the deterministic signal used for each.
#: Severity order is the author's judgement of damage to a live call, stated so it can
#: be argued with; it is not derived from data.
FAILURE_MODES = [
    ("tone_collapse_under_refusal",
     "More pressure markers after refusals than elsewhere, informal address after honorific, or repeated turns",
     1),
    ("numeral_currency_handling",
     "Amount introduced by the model, INR token, digits in a different script from the words, western grouping",
     2),
    ("language_mismatch",
     "Reply's matrix language (Hindi vs Marathi function words) differs from the borrower's",
     3),
    ("script_inconsistency",
     "Reply script differs from the borrower's romanised script, or switches within the call",
     4),
    ("register_drift_bucket",
     "Same first-turn pressure/courtesy across 5, 30 and 90 DPD (firmness does not track the bucket)",
     5),
    ("long_turns",
     "Replies over 60 words on a phone call",
     6),
]


def failure_counts(calls: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    all_turns = [t for c in calls for t in c["turns"]]
    n_turns = len(all_turns)
    tone = [c for c in calls if c["call"]["honorific_dropped"] or c["call"]["repeated_turns"]
            or (c["call"]["pressure_per_turn_after_refusal"] or 0) > (c["call"]["pressure_per_turn_other"] or 0)]
    num_turns = [t for t in all_turns if t["metrics"]["numerals"]["introduced_amounts"]
                 or t["metrics"]["numerals"]["currency"]["inr_token"]
                 or t["metrics"]["numerals"]["digit_script_mismatch"]
                 or t["metrics"]["numerals"]["western_grouping"]]
    script_turns = [t for t in all_turns if t["metrics"]["script"] not in ("latin", "none")]
    long_turns = [t for t in all_turns if t["metrics"]["long_turn"]]
    mismatch = [t for t in all_turns if t["metrics"]["language_mismatch"]]
    out["language_mismatch"] = {"turns": len(mismatch), "of_turns": n_turns,
                                "calls": sorted({c["call_id"] for c in calls
                                                 if c["call"]["language_mismatch_turns"]})}
    out["tone_collapse_under_refusal"] = {"calls": len(tone), "of_calls": len(calls),
                                          "call_ids": [c["call_id"] for c in tone]}
    out["numeral_currency_handling"] = {"turns": len(num_turns), "of_turns": n_turns}
    out["script_inconsistency"] = {"turns": len(script_turns), "of_turns": n_turns,
                                   "calls_switching": sum(1 for c in calls if c["call"]["script_switches"])}
    out["long_turns"] = {"turns": len(long_turns), "of_turns": n_turns}
    by_lang: dict[str, dict[int, Any]] = defaultdict(dict)
    for c in calls:
        ft = c["call"]["first_turn"]
        if ft:
            by_lang[c["language"]][c["dpd"]] = {"pressure": len(ft["pressure_markers"]),
                                                "courtesy": len(ft["courtesy_markers"]), "words": ft["words"]}
    flat = {lang: (len({(v["pressure"], v["courtesy"]) for v in d.values()}) == 1 and len(d) == len(BUCKETS))
            for lang, d in by_lang.items()}
    out["register_drift_bucket"] = {"first_turn_by_bucket": by_lang,
                                    "languages_with_identical_markers_across_buckets": sorted(k for k, v in flat.items() if v)}
    return out


def aggregate(calls: list[dict[str, Any]]) -> dict[str, Any]:
    all_turns = [t for c in calls for t in c["turns"]]
    scripts = defaultdict(lambda: defaultdict(int))
    for c in calls:
        for t in c["turns"]:
            scripts[c["language"]][t["metrics"]["script"]] += 1
    return {
        "calls": len(calls), "turns": len(all_turns),
        "script_by_language": {k: dict(v) for k, v in scripts.items()},
        "failure_modes": failure_counts(calls),
        "failure_mode_definitions": [{"mode": m, "signal": s, "severity_rank": r} for m, s, r in FAILURE_MODES],
        "per_call": {c["call_id"]: c["call"] for c in calls},
        "thinking_leaked": sum(1 for t in all_turns if t["thinking_leaked"]),
        "model_errors": sum(1 for t in all_turns if t["error"]),
        "latency_ttft_ms": latency_summary([t["ttft_ms"] for t in all_turns]),
        "latency_total_ms": latency_summary([t["total_ms"] for t in all_turns]),
        "note": "Text-side signals only. TTS survival requires audio; see tts.json.",
    }


# --- TTS / STT round trip -------------------------------------------------------------

class TtsEngine(Protocol):
    name: str

    def synthesize(self, text: str, language: str, out_path: Path) -> Path: ...


class SttEngine(Protocol):
    name: str

    def transcribe(self, audio_path: Path, language: str) -> str: ...


#: Engines the harness knows how to look for. Found = binary on PATH.
KNOWN_ENGINE_BINARIES = {"tts": ["piper", "espeak-ng", "espeak", "festival", "pico2wave"],
                         "stt": ["whisper", "whisper-cli", "whisper.cpp", "vosk-transcriber"]}


def detect_engines() -> dict[str, list[str]]:
    return {kind: [b for b in bins if shutil.which(b)] for kind, bins in KNOWN_ENGINE_BINARIES.items()}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", text)).strip()


def cer(reference: str, hypothesis: str) -> float | None:
    ref, hyp = _normalise(reference), _normalise(hypothesis)
    if not ref:
        return None
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return round(prev[-1] / len(ref), 4)


def roundtrip(text: str, language: str, tts: TtsEngine | None, stt: SttEngine | None,
              audio_path: Path) -> dict[str, Any]:
    """Speak ``text``, transcribe it back, compare. Records ``blocked`` without engines."""
    if tts is None:
        return {"status": "blocked", "reason": "no TTS engine available", "audio": None}
    audio = tts.synthesize(text, language, audio_path)
    if stt is None:
        return {"status": "audio_only", "reason": "no STT engine available", "audio": str(audio)}
    heard = stt.transcribe(audio, language)
    said_amounts = set(numeral_profile(text)["amounts"])
    heard_amounts = set(numeral_profile(heard)["amounts"])
    return {"status": "ok", "audio": str(audio), "tts": tts.name, "stt": stt.name,
            "transcript": heard, "cer": cer(text, heard),
            "amounts_said": sorted(said_amounts), "amounts_heard": sorted(heard_amounts),
            "amounts_preserved": said_amounts <= heard_amounts}


def tts_pass(calls: list[dict[str, Any]], tts: TtsEngine | None, stt: SttEngine | None,
             audio_dir: Path) -> dict[str, Any]:
    results = {}
    for c in calls:
        for t in c["turns"]:
            key = f"{c['call_id']}-t{t['turn']}"
            results[key] = roundtrip(t["reply"], c["language"], tts, stt, audio_dir / f"{key}.wav")
    statuses = defaultdict(int)
    for r in results.values():
        statuses[r["status"]] += 1
    return {"engines_found": detect_engines(), "status_counts": dict(statuses), "results": results}


# --- rubric and human rating ----------------------------------------------------------

#: Challenge 6.2 anchors are verbatim in ``challenge.REGISTER_RUBRIC``. The example
#: replies below are the author's illustrations for raters. They were written by the AI
#: assistant building this harness and have not been reviewed by a native speaker.
ANCHOR_EXAMPLES: dict[str, dict[str, str]] = {
    "naturalness": {
        "1": "Aadarniya grahak, aapke dwara deya rashi ka bhugtan abhi tak lambit hai.",
        "3": "Sunita ji, aapka two-wheeler loan ka payment 42,300 rupaye pending hai. Aap kab tak kar payengi?",
        "5": "Sunita ji, bas yaad dilane ke liye call kiya tha - 42,300 ka EMI pending hai. Kab tak ho payega?",
    },
    "code_mix_fit": {
        "1": "Sunita ji, your outstanding राशि 42,300 है, please kindly भुगतान करें immediately.",
        "3": "Sunita ji, aapka outstanding amount 42,300 rupaye hai, kripya shighra bhugtan karein.",
        "5": "Sunita ji, aapka 42,300 ka payment pending hai. Is hafte kab tak kar sakti hain?",
    },
    "bucket_fit": {
        "1": "(90 DPD) Koi baat nahi ji, jab time mile tab kar dijiyega, koi jaldi nahi hai.",
        "3": "(90 DPD) Sunita ji, payment 90 din se pending hai, please jaldi kar dijiye.",
        "5": "(90 DPD) Sunita ji, 90 din ho gaye hain aur 42,300 abhi bhi baaki hai. Aaj ek date tay kar lete hain - kaunsi date aapke liye possible hai?",
    },
    "tts_survival": {
        "1": "Aapka INR 42300.00 due hai (ref: ACC/2026/771) **turant** pay karein.",
        "3": "Aapka Rs. 42,300 due hai, please pay kar dijiye.",
        "5": "Aapka bayalis hazaar teen sau rupaye baaki hai.",
    },
    "consistency": {
        "1": "Turn 1: 'Sunita ji, aap...' -> Turn 6: 'Tum log paise kyun nahi dete? Ab dena padega.'",
        "3": "Turn 1 warm and polite; by turn 6 replies shorten and repeat 'payment kar dijiye' without acknowledging the borrower.",
        "5": "Turn 6 still acknowledges the refusal, keeps 'aap', and restates the ask once without pressure.",
    },
}


def rubric_document(verbatim: Any) -> dict[str, Any]:
    return {"source": "Challenge section 6.2 (anchors verbatim)", "scale": [1, 2, 3, 4, 5],
            "dimensions": verbatim, "anchor_examples": ANCHOR_EXAMPLES,
            "anchor_examples_status": "illustrative, AI-authored, not reviewed by a native speaker",
            "rating_unit": "one whole call (8 agent turns)",
            "tts_survival_status": "not ratable until audio exists; leave blank"}


def rating_template(calls_by_model: dict[str, list[dict[str, Any]]], seed: int = 20260930) -> tuple[list, dict]:
    items = []
    for model, calls in calls_by_model.items():
        for c in calls:
            items.append((model, c))
    random.Random(seed).shuffle(items)
    sheet, key = [], {}
    for n, (model, c) in enumerate(items, 1):
        rid = f"R{n:03d}"
        key[rid] = {"model": model, "call_id": c["call_id"]}
        sheet.append({"rating_id": rid, "language": c["language"], "dpd": c["dpd"],
                      "transcript": [{"borrower": t["borrower"], "agent": t["reply"]} for t in c["turns"]],
                      "scores": {"naturalness": None, "code_mix_fit": None, "bucket_fit": None,
                                 "tts_survival": None, "consistency": None},
                      "rater": None, "notes": ""})
    return sheet, key


def rater_agreement(sheets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    names = sorted(sheets)
    out: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa = {r["rating_id"]: r["scores"] for r in sheets[a]}
            sb = {r["rating_id"]: r["scores"] for r in sheets[b]}
            pair = {}
            for dim in ANCHOR_EXAMPLES:
                both = [(sa[k][dim], sb[k][dim]) for k in sa.keys() & sb.keys()
                        if isinstance(sa[k].get(dim), int) and isinstance(sb[k].get(dim), int)]
                pair[dim] = {"n": len(both),
                             "weighted_kappa": weighted_kappa([x for x, _ in both], [y for _, y in both],
                                                              [1, 2, 3, 4, 5]) if both else None}
            out[f"{a}~{b}"] = pair
    return out

