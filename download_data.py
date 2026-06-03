#!/usr/bin/env python3
"""Download the public Urdu-tweets emotion dataset.

Source: Bilal et al. (PeerJ CS, 2022), "Multi-label emotion classification of
Urdu tweets" — 6,043 Nastaliq tweets, 6 Ekman emotions, multi-label.
Repo:   https://github.com/Noman712/Mutilabel_Emotion_Detection_Urdu

Run this OUTSIDE any network sandbox:
    python download_data.py

It downloads the repo zip, lists every .csv/.xlsx/.tsv it contains with their
columns, and stages the most likely tweets+labels file at
    data/raw/urdu_emotion.csv
which `python main.py train --data data/raw/urdu_emotion.csv` can then use.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

REPO = "Noman712/Mutilabel_Emotion_Detection_Urdu"
RAW = Path(__file__).resolve().parent / "data" / "raw"
EMO_HINTS = {"anger", "disgust", "fear", "happiness", "happy", "joy",
             "sadness", "sad", "surprise"}
TEXT_HINTS = {"tweet", "text", "content", "sentence"}


def fetch_zip() -> zipfile.ZipFile:
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{REPO}/zip/refs/heads/{branch}"
        try:
            print(f"[download] {url}")
            req = Request(url, headers={"User-Agent": "urdu-emotion/0.1"})
            data = urlopen(req, timeout=60).read()
            return zipfile.ZipFile(io.BytesIO(data))
        except Exception as ex:  # noqa: BLE001
            print(f"  ! {branch}: {ex}")
    print("Could not download. Manually clone:\n"
          f"  git clone https://github.com/{REPO} data/raw/_repo")
    sys.exit(1)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    zf = fetch_zip()
    zf.extractall(RAW / "_repo")
    print(f"[extract] -> {RAW / '_repo'}")

    try:
        import pandas as pd
    except Exception:
        print("pandas not installed; install requirements first.")
        return

    candidates = []
    for path in sorted((RAW / "_repo").rglob("*")):
        if path.suffix.lower() not in (".csv", ".tsv", ".xlsx", ".xls"):
            continue
        try:
            df = (pd.read_excel(path) if path.suffix.lower().startswith(".xls")
                  else pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ","))
        except Exception as ex:  # noqa: BLE001
            print(f"  ? skip {path.name}: {ex}")
            continue
        cols = [str(c).strip().lower() for c in df.columns]
        score = sum(c in EMO_HINTS for c in cols) + sum(c in TEXT_HINTS for c in cols)
        print(f"  • {path.relative_to(RAW)}  rows={len(df)}  cols={list(df.columns)}")
        candidates.append((score, len(df), path, df))

    if not candidates:
        print("No tabular files found — inspect data/raw/_repo manually.")
        return

    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, _, best, df = candidates[0]
    out = RAW / "urdu_emotion.csv"
    df.to_csv(out, index=False)
    print(f"\n[staged] best guess -> {out}")
    print("Next:  python main.py prepare --data data/raw/urdu_emotion.csv")
    print("       python main.py train   --data data/raw/urdu_emotion.csv --epochs 40")


if __name__ == "__main__":
    main()
