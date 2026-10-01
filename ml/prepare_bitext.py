"""Build the 8-category training CSV from Bitext (+ ABCD for Technical and for style balance).

Run from the project root:

    python -m ml.prepare_bitext

Inputs (download once, see docs/bitext_mapping.md):
  data/raw/bitext/bitext_27k.csv       Bitext customer-support dataset (synthetic, 27 intents)
  data/raw/abcd/abcd_v1.1.json.gz      ABCD dialogues (a curated set of subflows, see ABCD_SUBFLOWS)
  --extra-technical file.csv           optional: more Technical tickets (column: message)

Output:
  data/raw/tickets_bitext.csv          columns: subject, message, category, source, intent, flags
  data/processed/bitext_prep_report.json

Then fine-tune with:
  python -m ml.train --data data/raw/tickets_bitext.csv --labels ml/labels_bitext8.json

Why ABCD: Bitext has no Technical (site/app bug) intents, so that class cannot be
trained from Bitext alone. ABCD rows are also added to other categories (capped) so the
writing style of ABCD is not a shortcut for "Technical". Nothing here is template-generated.
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import logging
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import BASE_DIR, MAX_MESSAGE_LEN, MIN_MESSAGE_LEN

logger = logging.getLogger("prepare_bitext")

CATEGORIES8: tuple[str, ...] = (
    "Account", "Payment", "Technical", "Refund", "Delivery", "Subscription", "General", "Other",
)

# Every one of the 27 Bitext intents is mapped explicitly. Unknown intents abort the run.
INTENT_TO_CATEGORY: dict[str, str] = {
    # Account: login, password, profile, account access
    "create_account": "Account",
    "delete_account": "Account",
    "edit_account": "Account",
    "switch_account": "Account",
    "recover_password": "Account",
    "registration_problems": "Account",
    # Payment: payment failed, charged, card/payment issues, invoices
    "payment_issue": "Payment",
    "check_payment_methods": "Payment",
    "check_invoice": "Payment",
    "get_invoice": "Payment",
    # Refund: refund request / status / policy
    "get_refund": "Refund",
    "track_refund": "Refund",
    "check_refund_policy": "Refund",
    # Delivery: tracking, delivery time/options, shipping address
    "track_order": "Delivery",
    "delivery_options": "Delivery",
    "delivery_period": "Delivery",
    "change_shipping_address": "Delivery",
    "set_up_shipping_address": "Delivery",
    # Subscription: newsletter / plan-related subscription requests
    "newsletter_subscription": "Subscription",
    "cancel_order": "Other",
    "check_cancellation_fee": "Subscription",
    # General: information, contact requests, feedback and service complaints
    "contact_customer_service": "General",
    "contact_human_agent": "General",
    "review": "General",
    "complaint": "General",
    # Other: order placement, changes and cancellations
    "place_order": "Other",
    "change_order": "Other",
}

def _cue(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_SHIP = r"ship|deliver|package|parcel|track|arriv|address|overnight|next week"
_REFUND = r"refund|money back|reimburs|credit"
_GENERAL_Q = r"\?|know|tell|explain|polic|price|cost|when|what|how|member|level|open|hour"

# (ABCD flow, subflow) -> (category, cue). The cue must appear in the customer message
# we keep, so vague openers ("I have a question") never get a specific label.
# Meanings were checked against sample conversations. Left OUT on purpose because they are
# ambiguous between our categories: shipping_issue/cost, subscription_inquiry/manage_pay_bill,
# manage_account/manage_change_address and status_*, product_defect/return_*.
ABCD_SUBFLOWS: dict[tuple[str, str], tuple[str, re.Pattern[str]]] = {
    # Technical: site faults only (search, cart, speed). App crashes/errors are NOT covered.
    ("troubleshoot_site", "search_results"): ("Technical", _cue(r"search|result|spinning|find")),
    ("troubleshoot_site", "shopping_cart"): ("Technical", _cue(r"cart|basket|checkout|add(ing|ed)? (it|an? |the )")),
    ("troubleshoot_site", "slow_speed"): ("Technical", _cue(r"slow|load|lag|speed|forever|freez|stuck|running|spinning")),
    # Payment
    ("troubleshoot_site", "credit_card"): ("Payment", _cue(r"card|credit|reject|declin|payment|purchase")),
    ("subscription_inquiry", "manage_dispute_bill"): ("Payment", _cue(r"charge|bill|dispute|fee|twice")),
    # Account
    ("account_access", "recover_password"): ("Account", _cue(r"password")),
    ("account_access", "recover_username"): ("Account", _cue(r"user ?name|log ?in|sign in|account")),
    ("account_access", "reset_2fa"): ("Account", _cue(r"2fa|two.?factor|authenticat|phone|code|verif")),
    ("manage_account", "manage_change_name"): ("Account", _cue(r"name")),
    ("manage_account", "manage_change_phone"): ("Account", _cue(r"phone|number")),
    # Refund
    ("product_defect", "refund_initiate"): ("Refund", _cue(_REFUND)),
    ("product_defect", "refund_status"): ("Refund", _cue(_REFUND)),
    ("product_defect", "refund_update"): ("Refund", _cue(_REFUND)),
    # Delivery (upgrade/downgrade here means changing the shipping speed or date)
    ("shipping_issue", "status"): ("Delivery", _cue(_SHIP)),
    ("shipping_issue", "missing"): ("Delivery", _cue(_SHIP)),
    ("shipping_issue", "manage"): ("Delivery", _cue(_SHIP)),
    ("order_issue", "manage_upgrade"): ("Delivery", _cue(_SHIP)),
    ("order_issue", "manage_downgrade"): ("Delivery", _cue(_SHIP)),
    # Subscription
    ("subscription_inquiry", "manage_extension"): ("Subscription", _cue(r"extension|extend|subscription|premium|membership")),
    # Other (adding/removing items on an existing order, like Bitext change_order)
    ("order_issue", "manage_cancel"): ("Other", _cue(r"order|item|remov|take off")),
    ("order_issue", "manage_create"): ("Other", _cue(r"order|item|add|cart")),
}
for _kind in ("membership", "policy", "pricing", "timing"):
    for _n in range(1, 5):
        ABCD_SUBFLOWS[("storewide_query", f"{_kind}_{_n}")] = ("General", _cue(_GENERAL_Q))

_GREETING = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening))\b[\w\s,!.]{0,12}$", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MAX_FOCUS_WORDS = 30

DEFAULT_BITEXT = BASE_DIR / "data" / "raw" / "bitext" / "bitext_27k.csv"
DEFAULT_ABCD = BASE_DIR / "data" / "raw" / "abcd" / "abcd_v1.1.json.gz"
DEFAULT_OUT = BASE_DIR / "data" / "raw" / "tickets_bitext.csv"
DEFAULT_REPORT = BASE_DIR / "data" / "processed" / "bitext_prep_report.json"
DEFAULT_SEED = 42
DEFAULT_PER_CLASS = 1500
DEFAULT_ABCD_PER_CLASS = 500
MIN_WORDS = 2
MAX_WORDS = 60
MIN_CLASS_ROWS = 100

_PLACEHOLDER = re.compile(r"\{\{\s*(.*?)\s*\}\}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SPACES = re.compile(r"\s+")

NAMES = ["Asha Rao", "Rahul Verma", "Priya Nair", "John Smith", "Maria Garcia", "Amit Shah", "Sara Khan"]
CITIES = ["Delhi", "Mumbai", "Pune", "London", "New York", "Toronto", "Berlin", "Sydney"]
COUNTRIES = ["India", "United States", "United Kingdom", "Canada", "Germany", "Australia"]
ACCOUNT_TYPES = ["premium", "basic", "business", "personal", "free", "pro"]
ACCOUNT_CATEGORIES = ["savings", "current", "business", "personal", "gold", "student"]
AMOUNTS = ["19.99", "49", "75.50", "120", "299", "499", "999", "1299", "2500"]
CURRENCIES = ["$", "\u20b9", "\u20ac", "\u00a3", "Rs. "]


# ---------------------------------------------------------------- text helpers
def clean_message(text: Any) -> str:
    """Unescape entities, drop control characters, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(unicodedata.normalize("NFKC", text))
    return _SPACES.sub(" ", _CONTROL.sub(" ", text)).strip()


def dedupe_key(text: str) -> str:
    """Key for duplicate detection.

    Lower-cases, masks placeholders and digits, drops punctuation and collapses
    repeated letters, so "cancel order #123" and "cancel oorder #987" collide.
    """
    t = _PLACEHOLDER.sub(" ph ", text.lower())
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"(.)\1+", r"\1", t)
    return _SPACES.sub(" ", t).strip()


def _placeholder_value(name: str, rng: random.Random) -> str | None:
    key = name.lower()
    if key == "order number":
        n = rng.randint(10000, 99999)
        return rng.choice([f"#{n}", str(n), f"ORD-{n}"])
    if key == "invoice number":
        return f"INV-{rng.randint(10000, 99999)}"
    if key == "person name":
        return rng.choice(NAMES)
    if key == "account type":
        return rng.choice(ACCOUNT_TYPES)
    if key == "account category":
        return rng.choice(ACCOUNT_CATEGORIES)
    if key == "refund amount":
        return rng.choice(AMOUNTS)
    if key == "currency symbol":
        return rng.choice(CURRENCIES)
    if key == "delivery city":
        return rng.choice(CITIES)
    if key == "delivery country":
        return rng.choice(COUNTRIES)
    return None


def fill_placeholders(text: str, rng: random.Random) -> str | None:
    """Replace {{Placeholders}} with realistic values; None if an unknown one is found."""
    unknown = False

    def repl(match: re.Match[str]) -> str:
        nonlocal unknown
        value = _placeholder_value(match.group(1), rng)
        if value is None:
            unknown = True
            return ""
        return value

    filled = _PLACEHOLDER.sub(repl, text)
    return None if unknown else _SPACES.sub(" ", filled).strip()


# ---------------------------------------------------------------- Bitext
def build_bitext_frame(raw: pd.DataFrame, rng: random.Random, stats: dict[str, Any]) -> pd.DataFrame:
    """Map, clean, de-duplicate and fill placeholders for the Bitext rows."""
    missing = {"instruction", "intent"} - set(raw.columns)
    if missing:
        raise SystemExit(f"Bitext CSV is missing columns: {sorted(missing)}")

    frame = pd.DataFrame({
        "template": raw["instruction"].map(clean_message),
        "intent": raw["intent"].astype(str).str.strip(),
        "flags": raw["flags"].astype(str) if "flags" in raw.columns else "",
    })
    unknown = sorted(set(frame["intent"]) - set(INTENT_TO_CATEGORY))
    if unknown:
        raise SystemExit(f"Unmapped Bitext intents (add them to INTENT_TO_CATEGORY): {unknown}")
    frame["category"] = frame["intent"].map(INTENT_TO_CATEGORY)
    stats["bitext_raw_rows"] = len(frame)

    frame = frame[frame["template"] != ""].copy()
    frame["key"] = frame["template"].map(dedupe_key)

    n_categories = frame.groupby("key")["category"].transform("nunique")
    stats["bitext_conflicting_rows_removed"] = int((n_categories > 1).sum())
    frame = frame[n_categories == 1]

    before = len(frame)
    frame = frame.drop_duplicates("key")
    stats["bitext_duplicates_removed"] = before - len(frame)

    filled = [fill_placeholders(t, rng) for t in frame["template"]]
    stats["bitext_unknown_placeholder_rows_removed"] = sum(f is None for f in filled)
    frame = frame.assign(message=filled).dropna(subset=["message"])

    words = frame["message"].str.split().str.len()
    length_ok = frame["message"].str.len().between(MIN_MESSAGE_LEN, MAX_MESSAGE_LEN)
    words_ok = words.between(MIN_WORDS, MAX_WORDS)
    stats["bitext_too_short_or_long_removed"] = int((~(length_ok & words_ok)).sum())
    frame = frame[length_ok & words_ok].copy()

    # Filling placeholders can make different templates identical (for example
    # "refund of {{Currency Symbol}}{{Refund Amount}}" and "refund of {{Order Number}}"),
    # so conflicts and duplicates are checked again on the FINAL text.
    frame["key"] = frame["message"].map(dedupe_key)
    n_categories = frame.groupby("key")["category"].transform("nunique")
    stats["bitext_conflicting_rows_removed_after_fill"] = int((n_categories > 1).sum())
    frame = frame[n_categories == 1]
    before = len(frame)
    frame = frame.drop_duplicates("key")
    stats["bitext_duplicates_removed_after_fill"] = before - len(frame)

    frame = frame.assign(source="bitext")
    return frame[["message", "category", "source", "intent", "flags", "key"]].reset_index(drop=True)


def cap_per_category(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Limit each category to ``cap`` rows, sampling evenly across its intents."""
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("category"):
        if len(group) <= cap:
            parts.append(group)
            continue
        sampled = group.groupby("intent", group_keys=False).sample(frac=cap / len(group), random_state=seed)
        if len(sampled) > cap:
            sampled = sampled.sample(n=cap, random_state=seed)
        parts.append(sampled)
    if not parts:
        return frame.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- ABCD (Technical)
def focus_sentences(text: str, cue: re.Pattern[str]) -> str:
    """Keep the sentence(s) that state the problem: drop greetings and 'my name is ...'
    and prefer sentences containing the cue. Keeps ABCD text close to ticket length."""
    sentences = [x.strip() for x in _SENTENCE_SPLIT.split(text) if x.strip()]
    sentences = [x for x in sentences if not _GREETING.match(x) and not re.match(r"my name is\b", x, re.IGNORECASE)]
    with_cue = [x for x in sentences if cue.search(x)]
    chosen = (with_cue or sentences)[:2]
    return " ".join(" ".join(chosen).split()[:MAX_FOCUS_WORDS])


def extract_abcd_rows(convos: list[dict[str, Any]]) -> pd.DataFrame:
    """One ticket per mapped conversation: the first customer message (of the first five)
    with >= 5 words that matches the subflow cue, reduced to its problem sentence(s)."""
    rows: list[dict[str, str]] = []
    for convo in convos:
        scenario = convo.get("scenario", {})
        subflow = scenario.get("subflow")
        mapped = ABCD_SUBFLOWS.get((scenario.get("flow"), subflow))
        if mapped is None:
            continue
        category, cue = mapped
        turns = [clean_message(text) for speaker, text in convo.get("original", []) if speaker == "customer"]
        for text in turns[:5]:
            if 5 <= len(text.split()) <= MAX_WORDS and cue.search(text):
                message = focus_sentences(text, cue)
                if message:
                    rows.append({
                        "message": message,
                        "category": category,
                        "intent": f"abcd:{scenario.get('flow')}/{subflow}",
                    })
                break
    return pd.DataFrame(rows, columns=["message", "category", "intent"])


def load_abcd_conversations(path: Path) -> list[dict[str, Any]]:
    """Read all splits of abcd_v1.1.json.gz into one list of conversations."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    return [convo for split in data.values() for convo in split]


def build_abcd_frame(convos: list[dict[str, Any]], stats: dict[str, Any]) -> pd.DataFrame:
    frame = extract_abcd_rows(convos)
    stats["abcd_candidates"] = len(frame)
    frame = frame[frame["message"].str.len().between(MIN_MESSAGE_LEN, MAX_MESSAGE_LEN)].copy()
    frame["key"] = frame["message"].map(dedupe_key)
    before = len(frame)
    frame = frame.drop_duplicates("key")
    stats["abcd_duplicates_removed"] = before - len(frame)
    return frame.assign(source="abcd", flags="")[
        ["message", "category", "source", "intent", "flags", "key"]
    ].reset_index(drop=True)


def build_extra_technical(extra: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    """Optional extra Technical tickets (CSV with a ``message`` column), e.g. app crashes,
    error messages and upload failures that ABCD does not cover."""
    if "message" not in extra.columns:
        raise SystemExit("--extra-technical CSV needs a 'message' column")
    frame = pd.DataFrame({"message": extra["message"].map(clean_message)})
    frame = frame[frame["message"].str.len().between(MIN_MESSAGE_LEN, MAX_MESSAGE_LEN)].copy()
    frame["key"] = frame["message"].map(dedupe_key)
    frame = frame.drop_duplicates("key")
    stats["extra_technical_rows"] = len(frame)
    return frame.assign(category="Technical", source="extra", intent="extra:technical", flags="")[
        ["message", "category", "source", "intent", "flags", "key"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------- assembly
def build_dataset(
    bitext_raw: pd.DataFrame,
    abcd_convos: list[dict[str, Any]],
    per_class: int = DEFAULT_PER_CLASS,
    seed: int = DEFAULT_SEED,
    abcd_per_class: int = DEFAULT_ABCD_PER_CLASS,
    extra_technical: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (final frame, stats). Pure function: no file IO."""
    stats: dict[str, Any] = {}
    rng = random.Random(seed)

    bitext = build_bitext_frame(bitext_raw, rng, stats)
    abcd = build_abcd_frame(abcd_convos, stats)

    clash = abcd["key"].isin(set(bitext["key"]))
    stats["abcd_rows_matching_bitext_removed"] = int(clash.sum())
    abcd = abcd[~clash]

    bitext = cap_per_category(bitext, per_class, seed)
    # Technical exists only in ABCD, so it may use up to per_class rows; every other
    # category gets at most abcd_per_class ABCD rows, so ABCD's writing style appears in
    # several categories and is not a shortcut for "Technical".
    technical = abcd[abcd["category"] == "Technical"]
    extra = pd.DataFrame(columns=technical.columns)
    if extra_technical is not None:
        extra = build_extra_technical(extra_technical, stats)
        extra = extra[~extra["key"].isin(set(bitext["key"]))]
        technical = technical[~technical["key"].isin(set(extra["key"]))]
    # Extra rows are curated by the user, so they are always kept; ABCD Technical fills the rest.
    room = max(per_class - len(extra), 0)
    technical = technical.sample(n=min(len(technical), room), random_state=seed)
    others = cap_per_category(abcd[abcd["category"] != "Technical"], abcd_per_class, seed)
    technical = pd.concat([extra, technical, others], ignore_index=True)

    final = pd.concat([bitext, technical], ignore_index=True)
    final = final.drop_duplicates("key")
    final = final.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    final.insert(0, "subject", "")
    final = final[["subject", "message", "category", "source", "intent", "flags"]]

    validate_output(final)
    stats.update(summarize(final))
    return final, stats


def validate_output(frame: pd.DataFrame) -> None:
    """Fail loudly if the dataset is not safe to fine-tune on."""
    unknown = set(frame["category"]) - set(CATEGORIES8)
    if unknown:
        raise SystemExit(f"Unknown categories in output: {sorted(unknown)}")
    counts = frame["category"].value_counts()
    thin = [c for c in CATEGORIES8 if counts.get(c, 0) < MIN_CLASS_ROWS]
    if thin:
        raise SystemExit(
            f"Categories with fewer than {MIN_CLASS_ROWS} rows: {thin}. "
            "Technical needs the ABCD file (see docs/bitext_mapping.md)."
        )
    if frame["message"].str.contains(r"\{\{|\}\}", regex=True).any():
        raise SystemExit("Unfilled placeholder left in a message")
    if frame["message"].map(dedupe_key).duplicated().any():
        raise SystemExit("Duplicates remain in the output")
    if not frame["message"].str.len().between(MIN_MESSAGE_LEN, MAX_MESSAGE_LEN).all():
        raise SystemExit("A message violates the app's length limits")


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    words = frame["message"].str.split().str.len()
    return {
        "final_rows": len(frame),
        "rows_per_category": {c: int((frame["category"] == c).sum()) for c in CATEGORIES8},
        "rows_per_source": {k: int(v) for k, v in frame["source"].value_counts().items()},
        "median_words_per_source": {k: float(v) for k, v in words.groupby(frame["source"]).median().items()},
        "rows_per_intent": {k: int(v) for k, v in frame["intent"].value_counts().items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the Bitext (+ABCD) 8-category dataset.")
    parser.add_argument("--bitext", type=Path, default=DEFAULT_BITEXT)
    parser.add_argument("--abcd", type=Path, default=DEFAULT_ABCD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--abcd-per-class", type=int, default=DEFAULT_ABCD_PER_CLASS)
    parser.add_argument("--extra-technical", type=Path, default=None,
                        help="optional CSV with a 'message' column of extra Technical tickets")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for path, hint in ((args.bitext, "Bitext CSV"), (args.abcd, "ABCD json.gz")):
        if not path.exists():
            raise SystemExit(f"{hint} not found at {path}. See docs/bitext_mapping.md for the download commands.")

    bitext_raw = pd.read_csv(args.bitext)
    convos = load_abcd_conversations(args.abcd)
    extra = pd.read_csv(args.extra_technical) if args.extra_technical else None
    final, stats = build_dataset(bitext_raw, convos, args.per_class, args.seed, args.abcd_per_class, extra)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out, index=False, encoding="utf-8")
    args.report.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in stats.items() if k != "rows_per_intent"}, indent=2))
    print(f"\nSaved {len(final)} rows -> {args.out}")
    print(f"Report -> {args.report}")
    print("Next: python -m ml.train --data data/raw/tickets_bitext.csv --labels ml/labels_bitext8.json")


if __name__ == "__main__":
    main()
