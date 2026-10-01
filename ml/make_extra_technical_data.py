"""Create a synthetic starter set for Technical tickets.

These generated examples improve coverage of app, upload, browser, sync, and
media failures, but do not replace real, anonymized customer tickets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.settings import BASE_DIR
from ml.prepare_bitext import dedupe_key

DEFAULT_OUT = BASE_DIR / "data" / "raw" / "extra_technical.csv"

ISSUES = (
    "the mobile app closes as soon as I open the home screen",
    "the app freezes when I tap the profile tab",
    "the settings page crashes before I can change a preference",
    "the app shuts down whenever I open the camera",
    "the screen becomes unresponsive after the latest app update",
    "the app gets stuck on its loading screen after I sign in",
    "the desktop app closes when I open a saved project",
    "the app crashes when I switch between two open documents",
    "the mobile app freezes while I scroll through older messages",
    "the app restarts every time I try to open the help section",
    "a photo upload fails even though the image is a supported format",
    "the upload progress bar reaches the end and then starts over",
    "the app reports an upload error for a small PDF file",
    "a video upload stops at the same point on every attempt",
    "the file picker opens but selecting a file does nothing",
    "the attachment disappears after the upload reports success",
    "downloaded reports arrive as zero-byte files",
    "the exported PDF contains blank pages instead of the report",
    "a CSV export is missing the final rows of my data",
    "the document download is cut off before the last page",
    "the website shows a blank page after I open the dashboard",
    "the page keeps spinning and never finishes loading",
    "I get a 500 error when I open the settings page",
    "the site returns a 502 error whenever I open a report",
    "the search page times out before displaying any results",
    "the page layout breaks and covers the save button",
    "the submit button stays disabled after I complete every field",
    "changes disappear when I refresh the preferences page",
    "the menu will not open in the latest version of my browser",
    "the page becomes unusably slow after I open a large table",
    "my desktop and phone show different versions of the same saved document",
    "new changes do not sync between the web app and my tablet",
    "the sync indicator spins forever without uploading my edits",
    "offline changes vanish after the app reconnects to the internet",
    "the app keeps creating duplicate copies while syncing a folder",
    "the dashboard data is stale even after I press refresh",
    "notifications stop appearing until I restart the app",
    "the notification sound plays but the notification itself is blank",
    "the app badge count stays unchanged after I read every alert",
    "scheduled reminders appear at the wrong time in the app",
    "video playback stops after a few seconds on a stable connection",
    "the audio player has no sound even though my device volume is on",
    "streaming video keeps buffering while other sites work normally",
    "the playback controls disappear when I rotate my phone",
    "captions drift out of sync with the video after a few minutes",
    "the preview pane stays black when I open an uploaded video",
    "the waveform never appears when I open an audio recording",
    "the application cannot detect my connected headset",
    "the map view remains empty after I enable location access",
    "the chart does not render after I choose a date range",
)

PREFIXES = (
    "",
    "I keep running into this: ",
    "Could you take a look? ",
    "Since yesterday, ",
    "I tried several times, but ",
    "This is still broken: ",
    "The same thing happens on every attempt: ",
    "No matter what I try, ",
)


def generate_messages() -> list[str]:
    """Return 400 deterministic, distinct synthetic technical messages."""
    messages = [f"{prefix}{issue}." for issue in ISSUES for prefix in PREFIXES]
    keys = [dedupe_key(message) for message in messages]
    if len(messages) != 400 or len(set(keys)) != len(keys):
        raise RuntimeError("Generated Technical messages must be 400 unique examples")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic extra Technical tickets.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"message": generate_messages()}).to_csv(
        args.out, index=False, encoding="utf-8"
    )
    print(f"Wrote {len(ISSUES) * len(PREFIXES)} synthetic Technical messages to {args.out}")
    print("Replace or supplement these with real, anonymized tickets before production use.")


if __name__ == "__main__":
    main()