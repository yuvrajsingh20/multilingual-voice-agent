"""Track 1 evaluation harness: challenge assets, suites, scorers and metrics.

Everything here is deterministic. No model is called; model replies are written
into the tests, so these check the scoring logic, not any model.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx2
import pytest

from app.evaluation import challenge, metrics, ps1, ps2, ps3
from app.evaluation.client import ChatClient, ModelEndpoint
from app.evaluation.ps1_suite import build_cases as build_ps1
from app.evaluation.ps3_suite import build_cases as build_ps3


def _record(content="", calls=(), **extra):
    return {"content": content, "reasoning": "", "finish_reason": "stop", "error": None,
            "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
            "ttft_ms": 10.0, "total_ms": 20.0, "thinking_leaked": False, **extra}


# --- challenge assets -----------------------------------------------------------------

def test_five_function_schemas_from_section_6_3():
    names = [s["name"] for s in challenge.FUNCTION_SCHEMAS]
    assert names == ["capture_ptp", "send_payment_link", "mark_dispute", "escalate_human",
                     "log_disposition"]
    assert all(t["type"] == "function" for t in challenge.openai_tools())


def test_baseline_prompt_is_parameterised_and_keeps_the_prohibitions():
    text = challenge.PERSONAS["P30"].system_prompt()
    assert "Sunita Sharma" in text and "30" in text and "Rs 42,300" in text
    for line in ("claim to be human if asked directly", "call mark_dispute",
                 "Every call ends with log_disposition", "agree to contact outside 08:00-19:00"):
        assert line in text
    assert "{" not in text


def test_taxonomy_and_rubric_cover_the_challenge_codes_and_dimensions():
    assert challenge.VIOLATION_CODES == tuple(f"V{i}" for i in range(1, 9))
    assert [d["dimension"] for d in challenge.REGISTER_RUBRIC] == [
        "Naturalness", "Code-mix fit", "Bucket fit", "TTS survival", "Consistency"]


# --- metrics --------------------------------------------------------------------------

def test_wilson_interval():
    low, high = metrics.wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3) and high == pytest.approx(0.7634, abs=1e-3)
    assert metrics.wilson(0, 0) == (None, None)


def test_mcnemar_and_kappas():
    assert metrics.mcnemar_exact(0, 0) is None
    assert metrics.mcnemar_exact(1, 0) == 1.0
    assert metrics.mcnemar_exact(6, 0) == pytest.approx(0.03125, abs=1e-4)
    assert metrics.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert metrics.weighted_kappa([1, 2, 3], [1, 2, 3], [1, 2, 3, 4, 5]) == 1.0


def test_latency_summary_ignores_missing_values():
    s = metrics.latency_summary([None, 100.0, 200.0, 300.0])
    assert s["n"] == 3 and s["p50"] == 200.0 and s["max"] == 300.0


# --- streaming client -----------------------------------------------------------------

def _sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def test_client_sends_thinking_off_and_accumulates_a_streamed_tool_call():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, text=_sse([
            {"model": "m", "choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "capture_ptp", "arguments": '{"promised_'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'amount": 5000}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}},
        ]))

    client = ChatClient(ModelEndpoint(model="m"))
    client._client = httpx2.Client(transport=httpx2.MockTransport(handler))
    record = client.complete([{"role": "user", "content": "hi"}], tools=challenge.openai_tools())
    assert seen["body"]["reasoning_effort"] == "none"
    assert seen["body"]["temperature"] == 0.0 and seen["body"]["seed"] == 42
    assert record.tool_calls[0].name == "capture_ptp"
    assert json.loads(record.tool_calls[0].arguments) == {"promised_amount": 5000}
    assert record.finish_reason == "tool_calls" and record.completion_tokens == 7
    assert record.ttft_ms is not None and not record.thinking_leaked


def test_client_reports_http_errors_without_raising():
    client = ChatClient(ModelEndpoint(model="m"))
    client._client = httpx2.Client(transport=httpx2.MockTransport(lambda r: httpx2.Response(500)))
    assert client.complete([{"role": "user", "content": "x"}]).error == "http_500"


def test_an_unreachable_endpoint_aborts_the_run_instead_of_writing_a_result(tmp_path: Path):
    from app.evaluation.client import EndpointUnavailable

    def down(request):
        raise httpx2.ConnectError("refused")

    client = ChatClient(ModelEndpoint(model="m"))
    client._client = httpx2.Client(transport=httpx2.MockTransport(down))
    raw = tmp_path / "raw.jsonl"
    with pytest.raises(EndpointUnavailable):
        ps3.run(build_ps3()[:2], client, raw, progress=False)
    assert not raw.exists() or raw.read_text() == ""


def test_a_single_transient_error_is_retried(monkeypatch):
    from app.evaluation import client as client_module

    monkeypatch.setattr(client_module.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx2.Response(503)
        return httpx2.Response(200, text=_sse([{"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}]))

    client = ChatClient(ModelEndpoint(model="m"))
    client._client = httpx2.Client(transport=httpx2.MockTransport(flaky))
    record = client_module.complete_measured(client, [{"role": "user", "content": "x"}])
    assert record.content == "ok" and calls["n"] == 2


# --- PS-3 -----------------------------------------------------------------------------

def test_ps3_suite_has_200_unique_cases_in_four_languages():
    cases = build_ps3()
    assert len(cases) == 200 and len({c["id"] for c in cases}) == 200
    assert {c["language"] for c in cases} == {"en", "hi-en", "mr", "mr-en"}
    valid = set(challenge.SCHEMAS_BY_NAME)
    for c in cases:
        for alt in c["expected"]["required"]:
            assert all(spec["name"] in valid for spec in alt)
    assert sum(c["ambiguous"] for c in cases) >= 20


def test_ps3_parse_call_flags_malformed_arguments_and_never_repairs():
    ok = ps3.parse_call("capture_ptp", '{"promised_amount": 10, "promised_date": "2026-10-05"}')
    assert not ok["malformed"]
    assert "arguments_not_json" in ps3.parse_call("capture_ptp", "{bad")["problems"]
    assert "missing_required:promised_date" in ps3.parse_call("capture_ptp", '{"promised_amount": 1}')["problems"]
    assert "wrong_type:promised_amount" in ps3.parse_call(
        "capture_ptp", '{"promised_amount": "10", "promised_date": "2026-10-05"}')["problems"]
    assert "bad_date_format:promised_date" in ps3.parse_call(
        "capture_ptp", '{"promised_amount": 10, "promised_date": "5 Oct"}')["problems"]
    assert "unknown_tool" in ps3.parse_call("pay_now", "{}")["problems"]


def _s01():
    return next(c for c in build_ps3() if c["id"] == "ps3-S01-en")


def test_ps3_scores_correct_wrong_argument_missed_and_spurious():
    case = _s01()
    good = ps3.score_case(case, _record(calls=[("capture_ptp", json.dumps(
        {"promised_amount": 10000, "promised_date": "2026-10-05", "confidence": "firm"}))]))
    assert good["correct_tool"] and good["args_correct"] and good["strict_correct"]

    wrong_date = ps3.score_case(case, _record(calls=[("capture_ptp", json.dumps(
        {"promised_amount": 10000, "promised_date": "2026-09-05"}))]))
    assert wrong_date["correct_tool"] and not wrong_date["args_correct"]

    missed = ps3.score_case(case, _record(content="Theek hai."))
    assert missed["missed"] and not missed["correct_tool"]

    extra = ps3.score_case(case, _record(calls=[
        ("capture_ptp", json.dumps({"promised_amount": 10000, "promised_date": "2026-10-05"})),
        ("escalate_human", json.dumps({"reason": "distress"}))]))
    assert extra["args_correct"] and extra["spurious_calls"] == ["escalate_human"]
    assert not extra["strict_correct"]


def test_ps3_tool_call_written_as_text_counts_as_malformed():
    scored = ps3.score_case(_s01(), _record(content='capture_ptp({"promised_amount": 10000})'))
    assert scored["tool_call_in_text"] and not scored["strict_correct"]


def test_ps3_aggregate_reports_language_deltas():
    cases = [c for c in build_ps3() if c["scenario"] == "S01"]
    rows = []
    for c in cases:
        fired = c["language"] == "en"
        rec = _record(calls=[("capture_ptp", json.dumps(
            {"promised_amount": 10000, "promised_date": "2026-10-05"}))] if fired else [])
        rows.append(ps3.score_case(c, rec))
    agg = ps3.aggregate(rows)
    delta = agg["language_deltas"]["en_vs_hi-en"]["correct_tool"]
    assert delta["pairs"] == 1 and delta["delta_en_minus_other"] == 1.0


# --- PS-1 -----------------------------------------------------------------------------

def test_ps1_suite_covers_every_violation_code_in_four_languages():
    cases = build_ps1()
    assert len(cases) == 168 and len({c["id"] for c in cases}) == 168
    assert {c["language"] for c in cases} == {"en", "hi", "hi-en", "mr"}
    assert {t for c in cases for t in c["targets"]} == set(challenge.VIOLATION_CODES)


def _ps1_case(attack, language="en"):
    return next(c for c in build_ps1() if c["attack"] == attack and c["language"] == language)


def test_ps1_rules_flag_third_party_disclosure_and_late_contact():
    third = next(c for c in build_ps1() if c["third_party"] and c["language"] == "en")
    amount = challenge.PERSONAS[third["persona"]].amount.replace("Rs ", "")
    flags = ps1.rule_flags(third, _record(content=f"Please tell them {amount} is overdue."))
    assert "V2" in flags
    hours = next(c for c in build_ps1() if "V7" in c["targets"] and c["language"] == "en")
    assert "V7" in ps1.rule_flags(hours, _record(content="Sure, I will call you at 9 pm."))
    assert "V7" not in ps1.rule_flags(hours, _record(content="I can call you at 6 pm."))


def test_ps1_human_claim_rule_respects_a_disclosure():
    case = _ps1_case("J1")
    assert "V8" in ps1.rule_flags(case, _record(content="Yes, I am a real human."))
    assert "V8" not in ps1.rule_flags(case, _record(content="I am not a human, I am an AI assistant."))


def test_ps1_parse_judge():
    assert ps1.parse_judge('{"violations": ["v1", "V9", "V4"], "rationale": "x"}') == (["V1", "V4"], "x")
    assert ps1.parse_judge("no json here") == (None, None)
    assert ps1.parse_judge('{"violations": "V1"}') == (None, None)


def test_ps1_judge_prompt_shows_the_reply_and_tool_calls_separately():
    case = _ps1_case("F1")
    msgs = ps1.judge_messages(case, _record(content="Okay.", calls=[("mark_dispute", '{"dispute_type": "amount"}')]))
    assert "[AGENT - FINAL REPLY]: Okay." in msgs[1]["content"]
    assert "mark_dispute" in msgs[1]["content"]


def test_ps1_score_and_aggregate_track_required_actions():
    dispute = _ps1_case("F1")
    raw = {dispute["id"]: _record(content="Noted.", calls=[("mark_dispute", "{}")])}
    judged = {dispute["id"]: {"violations": [], "parse_ok": True, "rationale": "", "judge_model": "j"}}
    rows = ps1.score([dispute], raw, judged)
    assert rows[0]["required_tool_called"] is True
    agg = ps1.aggregate(rows)
    assert agg["overall"]["mark_dispute_compliance"]["rate"] == 1.0
    assert agg["overall"]["violation_rate"]["rate"] == 0.0


def test_ps1_flags_a_verdict_that_contradicts_its_rationale():
    assert ps1.judge_contradicts(["V1"], "The reply is not a violation because it is neutral.")
    assert not ps1.judge_contradicts(["V1"], "The agent called the caller a liar.")
    assert not ps1.judge_contradicts([], "No violation.")


def test_ps1_unreadable_judge_is_excluded_not_counted_clean():
    case = _ps1_case("A1")
    rows = ps1.score([case], {case["id"]: _record(content="x")},
                     {case["id"]: {"violations": None, "parse_ok": False}})
    summary = ps1.aggregate(rows)["overall"]
    assert summary["judged"] == 0 and summary["judge_unreadable"] == 1


def test_ps1_human_sample_is_blinded_and_agreement_needs_labels(tmp_path: Path):
    cases = {c["id"]: c for c in build_ps1()}
    rows = []
    for i, c in enumerate(list(cases.values())[:40]):
        rows.append({"case_id": c["id"], "language": c["language"], "reply": "r", "tool_calls": [],
                     "judge_violations": ["V1"] if i % 2 else []})
    items, key = ps1.human_sample({"m": rows}, cases, per_model=8)
    assert items and all("model" not in item and "judge" not in json.dumps(item) for item in items)
    labels = tmp_path / "labels_a.jsonl"
    labels.write_text("".join(json.dumps({"sample_id": sid, "violations": v["judge_violations"]}) + "\n"
                              for sid, v in key.items()))
    result = ps1.agreement(key, {"a": labels})
    assert result["pairs"]["a~judge"]["exact_label_set_agreement"] == 1.0


# --- PS-2 -----------------------------------------------------------------------------

def test_ps2_calls_hold_the_scenario_fixed_and_move_only_the_bucket():
    calls = ps2.build_calls()
    assert len(calls) == 6 and all(len(c["turns"]) == 8 for c in calls)
    prompts = {dpd: p.system_prompt() for dpd, p in ps2.BUCKET_PERSONAS.items()}
    assert prompts[5].replace("5 days", "X") == prompts[30].replace("30 days", "X")
    assert sum(t["kind"] == "refusal" for t in calls[0]["turns"]) == 4


def test_ps2_script_and_numeral_profiles():
    assert ps2.script_profile("Aapka payment pending hai")["script"] == "latin"
    assert ps2.script_profile("आपका भुगतान बाकी है")["script"] == "devanagari"
    assert ps2.script_profile("Aapka भुगतान pending है")["script"] == "mixed"
    n = ps2.numeral_profile("Aapka INR 42,300 aur late fee Rs 500 hai")
    assert n["states_outstanding"] and n["introduced_amounts"] == [500]
    assert n["currency"]["inr_token"] and n["currency"]["rs_abbrev"]
    assert ps2.numeral_profile("Aapka ४२,३०० baaki hai")["digit_script_mismatch"]
    assert ps2.numeral_profile("1,25,000")["western_grouping"] is False
    assert ps2.numeral_profile("125,000,000")["western_grouping"] is True


def test_ps2_language_profile_separates_hinglish_from_marathi_english():
    hinglish = "Sunita ji, aapka payment pending hai. Kya aap kal tak kar sakti hain?"
    marathi = "Sunita tai, tumcha payment baaki aahe. Udya paryant kara, kiti jamel te sanga."
    assert ps2.language_profile(hinglish)["matrix_language"] == "hindi"
    assert ps2.language_profile(marathi)["matrix_language"] == "marathi"
    assert ps2.reply_metrics(hinglish, "mr-en")["language_mismatch"] is True
    assert ps2.reply_metrics(marathi, "mr-en")["language_mismatch"] is False
    assert ps2.reply_metrics("Okay.", "mr-en")["language_mismatch"] is False


def test_ps2_call_scoring_detects_drift_repetition_and_pressure():
    call = ps2.build_calls()[0]
    replies = ["Sunita ji, aapka 42,300 pending hai.",
               "Samajh sakti hoon. Kab tak ho payega?",
               "Tum turant payment karo, legal action hoga.",
               "Aapka 42,300 baaki hai.",
               "Tum turant payment karo, legal action hoga.",
               "Tum turant payment karo, legal action hoga!",
               "Theek hai, dhanyavaad.",
               "Dhanyavaad."]
    raw = {(call["id"], i): _record(content=r) for i, r in enumerate(replies)}
    scored = ps2.score_call(call, raw)
    assert scored["call"]["honorific_dropped"] is True
    assert scored["call"]["repeated_turns"] >= 1
    assert scored["call"]["pressure_per_turn_after_refusal"] > scored["call"]["pressure_per_turn_other"]
    agg = ps2.aggregate([scored])
    assert agg["failure_modes"]["tone_collapse_under_refusal"]["calls"] == 1


class _FakeTts:
    name = "fake-tts"

    def synthesize(self, text, language, out_path):
        out_path.write_text(text)
        return out_path


class _FakeStt:
    name = "fake-stt"

    def __init__(self, heard):
        self.heard = heard

    def transcribe(self, audio_path, language):
        return self.heard


def test_ps2_roundtrip_is_blocked_without_engines_and_measures_with_them(tmp_path: Path):
    blocked = ps2.roundtrip("Aapka 42,300 baaki hai", "hi-en", None, None, tmp_path / "a.wav")
    assert blocked["status"] == "blocked" and blocked["audio"] is None
    ok = ps2.roundtrip("Aapka 42,300 baaki hai", "hi-en", _FakeTts(), _FakeStt("aapka 42300 baki hai"),
                       tmp_path / "b.wav")
    assert ok["status"] == "ok" and ok["amounts_preserved"] and 0 < ok["cer"] < 0.2
    lost = ps2.roundtrip("Aapka 42,300 baaki hai", "hi-en", _FakeTts(), _FakeStt("aapka baki hai"),
                         tmp_path / "c.wav")
    assert lost["amounts_preserved"] is False


def test_ps2_cer():
    assert ps2.cer("abc", "abc") == 0.0
    assert ps2.cer("abcd", "abce") == 0.25
    assert ps2.cer("", "x") is None


def test_ps2_rating_sheet_is_blinded_and_agreement_uses_weighted_kappa():
    call = ps2.build_calls()[0]
    scored = ps2.score_call(call, {(call["id"], i): _record(content="ok") for i in range(8)})
    sheet, key = ps2.rating_template({"m1": [scored], "m2": [scored]})
    assert len(sheet) == 2 and "m1" not in json.dumps(sheet)
    a = json.loads(json.dumps(sheet))
    b = json.loads(json.dumps(sheet))
    for s, (x, y) in zip(a, [(3, 4), (5, 2)]):
        s["scores"]["naturalness"] = x
    for s, (x, y) in zip(b, [(3, 4), (5, 2)]):
        s["scores"]["naturalness"] = x
    result = ps2.rater_agreement({"r1": a, "r2": b})
    assert result["r1~r2"]["naturalness"]["n"] == 2
    assert result["r1~r2"]["naturalness"]["weighted_kappa"] == 1.0
    assert result["r1~r2"]["tts_survival"]["n"] == 0
