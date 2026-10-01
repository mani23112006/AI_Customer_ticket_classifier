"""Generate a SYNTHETIC sample dataset of support tickets.

*** SAMPLE DATA ONLY ***
Every row is machine-generated from templates with a fixed random seed. It is
for demonstrating the pipeline end to end. Metrics measured on it will look
better than they would on real customer tickets, because template-generated
text is much easier to separate than real-world text. Replace it with real,
labelled, consented data before drawing conclusions about model quality.

Run from the project root:  python -m ml.make_sample_data
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from config.settings import BASE_DIR, CATEGORIES

DEFAULT_SEED = 42
DEFAULT_PER_CLASS = 300
DEFAULT_OUT = BASE_DIR / "data" / "raw" / "tickets_sample.csv"

SUBJECTS: dict[str, list[str]] = {
    "Payment": [
        "Wrong charge on my bill", "Invoice question", "Charged twice", "Billing problem",
        "Payment deducted but not confirmed", "Unknown transaction on statement",
        "Extra tax on invoice", "Need copy of invoice",
    ],
    "Technical": [
        "App keeps crashing", "Error when uploading", "Website not loading", "Login screen freezes",
        "Notifications not working", "Sync problem", "Blank screen after update", "Feature not working",
    ],
    "Account": [
        "Cannot log in", "Password reset not working", "Change my email address", "Account locked",
        "Suspicious activity on account", "Update profile details", "Verification email missing",
        "Two-factor authentication help",
    ],
    "Refund": [
        "Refund request", "Where is my refund", "Money back please", "Refund not received",
        "Return and refund", "Wrong refund amount", "Refund pending for weeks", "Damaged item refund",
    ],
    "Delivery": [
        "Order not delivered", "Package is late", "Tracking not updating", "Delivered to wrong address",
        "Change delivery address", "Damaged package", "Delivery rescheduled again", "Where is my parcel",
    ],
    "Subscription": [
        "Cancel my subscription", "Please cancel order", "Stop auto renewal", "Cancellation request",
        "Cancel my membership", "Terminate my plan", "Cancel booking", "Cannot find cancel button",
    ],
    "General": [
        "Question about your service", "Support hours?", "Do you ship abroad", "Plan details needed",
        "Feedback for your team", "Complaint about your service", "Product launch question", "Student discount?",
        "Privacy policy question",
    ],
    "Other": [
        "Place a new order", "Change an item in my order", "Cancel my order", "Order help",
        "Add an item to my order", "Remove an item from my order", "Update my order", "Order request",
    ],
}

# Sentences per category. Some deliberately contain High-priority trigger words
# (hacked, unauthorized, stolen, deducted, fraud) so the priority engine has
# realistic input.
CORES: dict[str, list[str]] = {
    "Payment": [
        "I was charged {amount} twice for my {plan} plan this month.",
        "The invoice shows {amount} but my plan costs less than that.",
        "There is a wrong charge on my card statement from your company.",
        "My payment of {amount} was deducted but the invoice still shows unpaid.",
        "Money was deducted from my bank account but I did not get any receipt.",
        "Please explain the extra tax added to my latest bill.",
        "I need a copy of my invoice for {month}.",
        "My card was billed even after I updated my payment method.",
        "The discount code was not applied on my bill.",
        "I see an unknown charge of {amount} on my statement and I think it is fraud.",
        "How can I change the billing address on my invoices?",
        "Auto renewal took {amount} from my card without any notice.",
    ],
    "Technical": [
        "The app crashes every time I open the settings page.",
        "I get error code 500 when I try to upload a file.",
        "The website keeps loading forever and never opens.",
        "Push notifications stopped working after the latest update.",
        "I cannot connect the device to wifi, the setup keeps failing.",
        "The mobile app freezes on the login screen.",
        "The export to PDF button does nothing when I click it.",
        "Video playback keeps buffering even on a fast connection.",
        "The search feature returns no results even for obvious keywords.",
        "Bluetooth pairing fails with my {item}.",
        "I keep getting a timeout error while syncing my data.",
        "After the update the dashboard shows a blank white screen.",
    ],
    "Account": [
        "I forgot my password and the reset email never arrives.",
        "I want to change the email address linked to my account.",
        "My account is locked after too many login attempts.",
        "Someone hacked my account and changed my password.",
        "I noticed unauthorized logins from a device I do not recognize.",
        "Please update the phone number on my profile.",
        "I cannot verify my email because the verification link has expired.",
        "How do I turn on two-factor authentication for my account?",
        "I have two accounts and want them merged into one.",
        "My username shows the wrong name, please correct it.",
        "I think my account was compromised, I see activity that was not me.",
    ],
    "Refund": [
        "I want a refund for my {item}, it arrived damaged.",
        "Please refund {amount}, I was charged for a service I never used.",
        "It has been {days} days and my refund has not reached my bank account.",
        "I returned the {item} last week, when will I get my money back?",
        "The refund amount is less than what I originally paid.",
        "I am requesting a full refund because the product does not match the description.",
        "The refund status still shows pending for order {order}.",
        "I received the wrong {item} and I want my money back.",
        "Please process the refund to my original payment method.",
        "An unauthorized transaction of {amount} appeared and I need it refunded.",
    ],
    "Delivery": [
        "My order {order} has not arrived and it is {days} days late.",
        "The tracking page says delivered but I never received my package.",
        "The courier left my parcel at the wrong address.",
        "I want to change the delivery address for order {order}.",
        "The package arrived with the box torn open.",
        "Delivery keeps getting rescheduled without any explanation.",
        "My {item} was stolen from my doorstep right after delivery.",
        "There has been no tracking update for my shipment in {days} days.",
        "Can I choose a different delivery time slot for tomorrow?",
        "The delivery agent was rude and did not call before arriving.",
        "Order {order} shows shipped but nothing has moved.",
    ],
    "Subscription": [
        "I want to cancel my {plan} subscription effective immediately.",
        "Please cancel order {order}, I ordered it by mistake.",
        "How do I stop the auto renewal of my plan?",
        "I would like to cancel my membership before the next billing date.",
        "Cancel my subscription please, I no longer need the service.",
        "I changed my mind and want to cancel my order before it ships.",
        "Kindly confirm the cancellation of my {plan} plan in writing.",
        "Please terminate my contract at the end of this month.",
        "I tried to cancel in the app but the cancel button is missing.",
        "Cancel my booking {order} and confirm by email.",
    ],
    "General": [
        "What are your customer support working hours?",
        "Do you ship internationally and what are the rates?",
        "Can you tell me more about the features of the {plan} plan?",
        "Where can I find your terms and conditions?",
        "Do you offer discounts for students or bulk orders?",
        "Is there a physical store I can visit?",
        "How can I reach a manager to share feedback?",
        "Do you have a mobile app for both Android and iOS?",
        "I would like to know when the new product line will launch.",
        "Can I get a demo of your service before buying?",
        "What is your policy on data privacy?",
        "I have a suggestion to improve your website.",
        "I want to make a complaint about the service I received.",
        "I am unhappy with your service and would like to file a complaint.",
    ],
    "Other": [
        "I want to place a new order for a few items.",
        "Please add this item to my existing order.",
        "I need to remove one item from the order I just placed.",
        "I want to change the items in my order before it ships.",
        "Please help me place an order for this product.",
        "There is a problem with the items in my order.",
        "I need to cancel the order I just placed.",
    ],
}

OPENERS: dict[str, list[str]] = {
    "polite": ["Hello,", "Hi team,", "Dear support,", "Good morning,", "Hello, hope you are well."],
    "angry": [
        "This is unacceptable.", "I am really frustrated.", "I am very disappointed with your service.",
        "This is the third time I am writing.", "Honestly this is ridiculous.",
    ],
    "casual": ["hey", "hi there", "yo", "hi,", "hey guys"],
    "neutral": [""],
}
CLOSERS = [
    "Please help.", "Thanks.", "Waiting for your reply.", "Regards.", "Please respond soon.",
    "Thank you for your time.", "Let me know what to do.", "Need this fixed asap.",
]
FILLERS = [
    "I have been a customer for two years.", "I already tried contacting you earlier.",
    "I am writing from my phone.", "This happened yesterday evening.",
    "I checked the help page but could not find an answer.",
]
ITEMS = ["laptop", "headphones", "phone case", "jacket", "router", "keyboard", "watch", "backpack"]
PLANS = ["Pro", "Premium", "Basic", "Annual", "Family"]
AMOUNTS = ["$29.99", "$120", "$49", "Rs. 499", "Rs. 1299", "Rs. 2500", "$9.99", "$75.50"]
MONTHS = ["January", "March", "May", "July", "September", "November"]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        amount=rng.choice(AMOUNTS),
        item=rng.choice(ITEMS),
        order=f"#{rng.randint(10000, 99999)}",
        plan=rng.choice(PLANS),
        days=rng.randint(2, 14),
        month=rng.choice(MONTHS),
    )


def add_typos(text: str, rng: random.Random, rate: float = 0.06) -> str:
    """Introduce realistic character-level typos in a small share of words."""
    out: list[str] = []
    for word in text.split(" "):
        if len(word) > 3 and rng.random() < rate:
            i = rng.randrange(1, len(word) - 1)
            kind = rng.choice(["swap", "drop", "dup"])
            if kind == "swap":
                word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            elif kind == "drop":
                word = word[:i] + word[i + 1:]
            else:
                word = word[:i] + word[i] + word[i:]
        out.append(word)
    return " ".join(out)


def _make_message(category: str, rng: random.Random) -> str:
    tone = rng.choice(list(OPENERS))
    parts: list[str] = []
    opener = rng.choice(OPENERS[tone])
    if opener:
        parts.append(opener)
    n_cores = rng.choices([1, 2, 3], weights=[5, 4, 1])[0]
    parts += [_fill(c, rng) for c in rng.sample(CORES[category], k=n_cores)]
    if rng.random() < 0.35:
        parts.append(rng.choice(FILLERS))
    if rng.random() < 0.07:  # light cross-topic noise so classes are not perfectly clean
        other = rng.choice([c for c in CATEGORIES if c != category])
        parts.append(_fill(rng.choice(CORES[other]), rng))
    if rng.random() < 0.6:
        parts.append(rng.choice(CLOSERS))
    text = " ".join(parts)
    roll = rng.random()
    if roll < 0.15:
        text = text.lower()
    elif roll < 0.18:
        text = text.upper()
    if rng.random() < 0.30:
        text = add_typos(text, rng)
    return text


def _make_subject(category: str, rng: random.Random) -> str:
    subject = rng.choice(SUBJECTS[category])
    if rng.random() < 0.2:
        subject = subject.lower()
    if rng.random() < 0.15:
        subject = add_typos(subject, rng, rate=0.15)
    return subject


def generate_dataset(per_class: int = DEFAULT_PER_CLASS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return a balanced, shuffled, de-duplicated synthetic dataset."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for category in CATEGORIES:
        seen: set[tuple[str, str]] = set()
        attempts = 0
        while len(seen) < per_class and attempts < per_class * 100:
            attempts += 1
            subject = _make_subject(category, rng)
            message = _make_message(category, rng)
            key = (subject.lower(), message.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {"subject": subject, "message": message, "category": category, "source": "synthetic_sample"}
            )
        if len(seen) < per_class:
            raise RuntimeError(f"Could only generate {len(seen)} unique rows for {category}")
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


NOTICE = (
    "SAMPLE DATA ONLY\n"
    "tickets_sample.csv is synthetic (template-generated, fixed seed). It exists to demo the\n"
    "pipeline. Do not report model quality from it as if it were real-world performance.\n"
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sample tickets.")
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.per_class < 200:
        parser.error("--per-class must be at least 200")

    frame = generate_dataset(args.per_class, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, encoding="utf-8")
    (args.out.parent / "SAMPLE_DATA_NOTICE.txt").write_text(NOTICE, encoding="utf-8")
    print(f"Wrote {len(frame)} rows to {args.out}")
    print(frame["category"].value_counts().to_string())
    print("Reminder: this is SAMPLE (synthetic) data only.")


if __name__ == "__main__":
    main()
