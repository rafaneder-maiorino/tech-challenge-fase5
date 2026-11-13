"""Dataset audit for the Phase 5 Datathon (D1).

Downloads the candidate CSVs from Hugging Face, records their SHA-256, and
checks the acceptance criteria from the plan (section 4.1):

1. Size, languages, null rates, duplicates.
2. Label space: how many queues, how imbalanced, which ones count as "technical".
3. Signal: majority baseline vs TF-IDF + LogisticRegression (stratified 5-fold,
   macro F1) on subject+body, English only, consolidated classes.
4. Leakage probe: columns we must NOT use as features (answer, tags) are checked
   for how much of the label they give away.

Writes a Markdown report to reports/dataset_audit.md.

Usage:
    uv run python scripts/dataset_audit.py
    uv run python scripts/dataset_audit.py --min-class-size 300 --lang en
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

logger = logging.getLogger("dataset_audit")

REPO = "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main"
FILES = {
    "v5_multi": "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    "v4_multi_20k": "dataset-tickets-multi-lang-4-20k.csv",
}
# Queues that should trigger the RAG path. Adjust after reading the report.
TECHNICAL_QUEUES = {
    "Technical Support",
    "IT Support",
    "Product Support",
    "Service Outages and Maintenance",
}
SEED = 42


def download(name: str, cache_dir: Path) -> tuple[pd.DataFrame, str]:
    """Download (or reuse cached) CSV and return it with its SHA-256."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / FILES[name]
    if not path.exists():
        url = f"{REPO}/{FILES[name]}"
        logger.info("downloading %s", url)
        with urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed HF URL
            path.write_bytes(resp.read())
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(io.BytesIO(raw))
    logger.info("%s: %d rows, sha256=%s", name, len(df), sha[:12])
    return df, sha


def build_text(df: pd.DataFrame) -> pd.Series:
    """Feature text = subject + body. Never the agent answer or the tags."""
    subject = df.get("subject", pd.Series("", index=df.index)).fillna("")
    body = df.get("body", pd.Series("", index=df.index)).fillna("")
    return (subject.astype(str) + " \n " + body.astype(str)).str.strip()


def consolidate(labels: pd.Series, min_size: int) -> pd.Series:
    """Keep queues with >= min_size rows; everything else becomes 'Other'."""
    counts = labels.value_counts()
    keep = set(counts[counts >= min_size].index)
    return labels.where(labels.isin(keep), "Other")


def cv_macro_f1(texts: pd.Series, y: pd.Series, model) -> tuple[float, float]:
    """Stratified 5-fold macro F1; NaN if the text column carries no vocabulary."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    try:
        scores = cross_val_score(model, texts, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    except ValueError as exc:  # e.g. empty vocabulary on a mostly-null column
        logger.warning("cv skipped: %s", exc)
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


def leakage_probe(df: pd.DataFrame, y: pd.Series, column: str) -> float | None:
    """Macro F1 when predicting the label from a column we will NOT use."""
    if column not in df.columns:
        return None
    x = df[column].fillna("").astype(str)
    if x.str.len().sum() == 0:
        return None
    model = make_pipeline(TfidfVectorizer(min_df=2), LogisticRegression(max_iter=2000))
    return cv_macro_f1(x, y, model)[0]


def audit(name: str, df: pd.DataFrame, sha: str, lang: str, min_size: int) -> list[str]:
    out: list[str] = [f"## `{FILES[name]}`", "", f"- SHA-256: `{sha}`", f"- Rows: {len(df):,}"]
    out.append(f"- Columns: {', '.join(df.columns)}")

    nulls = (df.isna().mean() * 100).round(1)
    out.append("- Null %: " + ", ".join(f"{c}={v}" for c, v in nulls.items() if v > 0))

    if "language" in df.columns:
        langs = df["language"].value_counts()
        out.append("- Languages: " + ", ".join(f"{k}={v:,}" for k, v in langs.items()))

    df = df.copy()
    df["text"] = build_text(df)
    dup = int(df["text"].duplicated().sum())
    out.append(f"- Exact duplicate texts (subject+body): {dup:,} ({dup / len(df):.1%})")

    sub = df[df["language"] == lang] if "language" in df.columns else df
    sub = sub[sub["text"].str.len() > 0].drop_duplicates("text")
    out.append(f"- Usable rows ({lang}, non-empty, deduplicated): {len(sub):,}")

    lengths = sub["text"].str.len()
    out.append(
        "- Text length (chars): "
        f"p50={lengths.median():.0f}, p95={lengths.quantile(0.95):.0f}, "
        f"max={lengths.max():.0f}, >1000={(lengths > 1000).mean():.1%}"
    )

    queues = sub["queue"].fillna("<null>")
    out += [
        "",
        f"### Queues ({queues.nunique()} distinct, {lang})",
        "",
        "| queue | n | % | technical? |",
        "|---|---|---|---|",
    ]
    for q, n in queues.value_counts().items():
        out.append(f"| {q} | {n:,} | {n / len(sub):.1%} | {'yes' if q in TECHNICAL_QUEUES else ''} |")

    y = consolidate(queues, min_size)
    tech_share = queues.isin(TECHNICAL_QUEUES).mean()
    out += [
        "",
        f"- Classes after consolidation (min {min_size}): {y.nunique()} → {', '.join(sorted(y.unique()))}",
        f"- Share routed to RAG (technical): {tech_share:.1%}",
    ]

    if "type" in sub.columns:
        out.append("- type: " + ", ".join(f"{k}={v:,}" for k, v in sub["type"].value_counts().items()))

    # Signal
    texts = sub["text"]
    majority = cv_macro_f1(texts, y, DummyClassifier(strategy="most_frequent"))
    tfidf_lr = cv_macro_f1(
        texts,
        y,
        make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
    )
    subject_only = cv_macro_f1(
        sub["subject"].fillna("").astype(str),
        y,
        make_pipeline(TfidfVectorizer(min_df=2), LogisticRegression(max_iter=2000, class_weight="balanced")),
    )
    tech_binary = cv_macro_f1(
        texts,
        queues.isin(TECHNICAL_QUEUES),
        make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
    )
    out += [
        "",
        "### Signal (stratified 5-fold, macro F1)",
        "",
        "| model | macro F1 |",
        "|---|---|",
        f"| majority class | {majority[0]:.3f} ± {majority[1]:.3f} |",
        f"| TF-IDF(1,2) + LR, subject+body | **{tfidf_lr[0]:.3f} ± {tfidf_lr[1]:.3f}** |",
        f"| TF-IDF + LR, subject only | {subject_only[0]:.3f} ± {subject_only[1]:.3f} |",
        f"| TF-IDF + LR, technical vs not (binary) | {tech_binary[0]:.3f} ± {tech_binary[1]:.3f} |",
    ]

    # Leakage probe on columns excluded from features
    out += [
        "",
        "### Leakage probe (columns NOT used as features)",
        "",
        "| column | macro F1 predicting queue |",
        "|---|---|",
    ]
    for col in ["answer", "tag_1", "tag_2", "priority"]:
        score = leakage_probe(sub, y, col)
        out.append(f"| {col} | {'n/a' if score is None else f'{score:.3f}'} |")

    gap = tfidf_lr[0] - majority[0]
    verdict = "PASS" if gap >= 0.15 else "WEAK SIGNAL — investigate before using"
    out += ["", f"**Signal verdict:** {verdict} (gap over majority = {gap:+.3f})", ""]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", default="en")
    parser.add_argument("--min-class-size", type=int, default=300)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("reports/dataset_audit.md"))
    parser.add_argument("--files", nargs="+", default=list(FILES), choices=list(FILES))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    lines = [
        "# Dataset audit — Tobi-Bueck/customer-support-tickets",
        "",
        "License: CC BY-NC 4.0 (non-commercial; OK for academic use, cite the source).",
        f"Language filter: `{args.lang}` · min class size: {args.min_class_size} · seed: {SEED}",
        "",
    ]
    for name in args.files:
        df, sha = download(name, args.cache_dir)
        lines += audit(name, df, sha, args.lang, args.min_class_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    logger.info("report written to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
