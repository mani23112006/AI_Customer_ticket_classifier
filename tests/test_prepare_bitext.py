"""Tests for the Bitext/ABCD dataset builder (no raw data needed)."""
import random

import pandas as pd
import pytest

from ml.prepare_bitext import (
    ABCD_SUBFLOWS,
    CATEGORIES8,
    INTENT_TO_CATEGORY,
    build_bitext_frame,
    build_dataset,
    cap_per_category,
    clean_message,
    dedupe_key,
    extract_abcd_rows,
    fill_placeholders,
    focus_sentences,
)
from ml.make_extra_technical_data import generate_messages

BITEXT_INTENTS = {
    "check_invoice", "complaint", "contact_customer_service", "edit_account", "switch_account",
    "check_payment_methods", "contact_human_agent", "delivery_period", "get_invoice",
    "newsletter_subscription", "payment_issue", "registration_problems", "cancel_order", "place_order",
    "track_refund", "change_order", "check_refund_policy", "create_account", "get_refund", "review",
    "set_up_shipping_address", "delete_account", "delivery_options", "recover_password", "track_order",
    "change_shipping_address", "check_cancellation_fee",
}


def test_all_27_bitext_intents_are_mapped_to_valid_categories():
    assert set(INTENT_TO_CATEGORY) == BITEXT_INTENTS
    assert set(INTENT_TO_CATEGORY.values()) <= set(CATEGORIES8)
    assert "Technical" not in INTENT_TO_CATEGORY.values()  # Bitext has no Technical intents
    assert INTENT_TO_CATEGORY["cancel_order"] == "Other"
    assert INTENT_TO_CATEGORY["complaint"] == "General"
    assert INTENT_TO_CATEGORY["review"] == "General"


def test_abcd_map_targets_are_valid():
    assert {cat for cat, _ in ABCD_SUBFLOWS.values()} <= set(CATEGORIES8)
    assert ("troubleshoot_site", "credit_card") in ABCD_SUBFLOWS
    assert ABCD_SUBFLOWS[("troubleshoot_site", "credit_card")][0] == "Payment"


def test_dedupe_key_collapses_typos_digits_and_placeholders():
    assert dedupe_key("Cancel order #123!") == dedupe_key("cancel oorder 987")
    assert dedupe_key("refund {{Order Number}}") != dedupe_key("refund of my order")


def test_fill_placeholders_leaves_no_braces_and_flags_unknown():
    rng = random.Random(1)
    filled = fill_placeholders("refund {{Currency Symbol}}{{Refund Amount}} for {{Order Number}}", rng)
    assert filled is not None and "{{" not in filled
    assert fill_placeholders("hello {{Something New}}", rng) is None


def test_clean_message_normalises_whitespace_and_entities():
    assert clean_message("  a&amp;b \n\t c ") == "a&b c"
    assert clean_message(None) == ""


WORDS = [
    "apple", "river", "cloud", "tiger", "maple", "stone", "ocean", "piano", "lemon", "amber",
    "cactus", "falcon", "garden", "harbor", "island", "jungle", "kettle", "lantern", "meadow", "nickel",
    "orchid", "pepper", "quartz", "rocket", "silver", "tunnel", "violet", "walnut", "yellow", "zebra",
]


def _bitext_rows():
    rows = []
    for intent in INTENT_TO_CATEGORY:
        for word in WORDS:
            rows.append({"flags": "B", "instruction": f"please help with {intent.replace('_', ' ')} for {word} today", "intent": intent})
    rows.append({"flags": "B", "instruction": "please help with get refund for apple today", "intent": "get_refund"})  # duplicate
    return pd.DataFrame(rows)


def _slow_convos(n):
    return [_convo("troubleshoot_site", "slow_speed", [f"the website about {WORDS[i % 30]}{'s' * (i // 30)} is running really slow today"]) for i in range(n)]


def test_build_bitext_frame_dedupes_and_maps():
    stats: dict = {}
    frame = build_bitext_frame(_bitext_rows(), random.Random(0), stats)
    assert stats["bitext_duplicates_removed"] >= 1
    assert set(frame["category"]) <= set(CATEGORIES8)
    assert not frame["key"].duplicated().any()


def test_unknown_intent_aborts():
    raw = pd.DataFrame({"flags": ["B"], "instruction": ["some text here"], "intent": ["brand_new_intent"]})
    with pytest.raises(SystemExit):
        build_bitext_frame(raw, random.Random(0), {})


def test_cap_per_category_spreads_across_intents():
    frame = pd.DataFrame({
        "category": ["A"] * 100, "intent": ["i1"] * 50 + ["i2"] * 50, "message": [str(i) for i in range(100)],
    })
    capped = cap_per_category(frame, 20, seed=1)
    assert len(capped) <= 20
    assert set(capped["intent"]) == {"i1", "i2"}


def test_focus_sentences_drops_greeting_and_name():
    cue = ABCD_SUBFLOWS[("troubleshoot_site", "slow_speed")][1]
    out = focus_sentences("Hi. My name is Crystal Minh. The website is very slow today.", cue)
    assert out == "The website is very slow today."


def _convo(flow, subflow, customer_turns):
    return {"scenario": {"flow": flow, "subflow": subflow},
            "original": [["agent", "Hi!"]] + [["customer", t] for t in customer_turns]}


def test_extract_abcd_skips_vague_openers_and_unmapped_subflows():
    convos = [
        _convo("troubleshoot_site", "slow_speed", ["Hello", "your website is running really slow right now"]),
        _convo("troubleshoot_site", "credit_card", ["I have a question about my account", "my credit card keeps getting rejected at checkout"]),
        _convo("shipping_issue", "cost", ["my shipping cost was way too high on this order"]),  # not mapped
        _convo("troubleshoot_site", "search_results", ["Hi", "ok"]),  # never states the problem
    ]
    rows = extract_abcd_rows(convos)
    assert sorted(rows["category"]) == ["Payment", "Technical"]


def test_build_dataset_end_to_end_small(monkeypatch):
    monkeypatch.setattr("ml.prepare_bitext.MIN_CLASS_ROWS", 5)  # tiny fixture, real default is 100
    bitext = _bitext_rows()
    convos = _slow_convos(30)
    final, stats = build_dataset(bitext, convos, per_class=25, seed=3, abcd_per_class=10)
    assert set(final["category"]) == set(CATEGORIES8)
    assert final["category"].value_counts().min() >= 5
    assert not final["message"].str.contains(r"\{\{").any()
    assert list(final.columns) == ["subject", "message", "category", "source", "intent", "flags"]
    again, _ = build_dataset(bitext, convos, per_class=25, seed=3, abcd_per_class=10)
    assert final.equals(again)  # deterministic


def test_extra_technical_rows_are_added(monkeypatch):
    monkeypatch.setattr("ml.prepare_bitext.MIN_CLASS_ROWS", 5)
    bitext = _bitext_rows()
    convos = _slow_convos(30)
    extra = pd.DataFrame({"message": ["the mobile app crashes when I open settings", "error 500 while uploading my file"]})
    final, stats = build_dataset(bitext, convos, per_class=25, seed=3, abcd_per_class=10, extra_technical=extra)
    assert stats["extra_technical_rows"] == 2
    assert (final["source"] == "extra").sum() == 2


def test_extra_technical_starter_messages_are_varied_and_unique():
    messages = generate_messages()
    assert len(messages) == 400
    assert len({dedupe_key(message) for message in messages}) == 400
    assert any("upload" in message for message in messages)
    assert any("sync" in message for message in messages)
    assert any("crashes" in message or "closes" in message for message in messages)


def test_thin_class_is_rejected():
    from ml.prepare_bitext import validate_output

    frame = pd.DataFrame({
        "subject": [""] * 3, "message": ["please help me with this"] * 3, "category": ["Account"] * 3,
        "source": ["bitext"] * 3, "intent": ["x"] * 3, "flags": [""] * 3,
    })
    with pytest.raises(SystemExit):
        validate_output(frame)
