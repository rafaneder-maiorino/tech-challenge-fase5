"""Clean the raw tickets and write a paraphrase-aware train/test split plus the German drift pool.

Why the split is grouped: the dataset contains near-duplicate paraphrases. With a plain random
split, 57.4% of test tickets had a train neighbour with TF-IDF cosine >= 0.7, and that neighbour
carried the same queue 98.3-100% of the time, so the model was scored on memory, not on
generalisation (docs/findings.md). Tickets are therefore grouped into paraphrase families
(connected components of the cosine >= 0.7 graph) and whole families go to one side only.
The same groups drive the grouped cross-validation in training.

Outputs (data/processed/ is gitignored, reproduced by `make prepare`):
    train.parquet, test.parquet   English, ~80/20 StratifiedGroupKFold fold, seed 42, with group_id
    drift_pool_de.parquet         German tickets, same cleaning; drift scenario for D8
    reports/prepare_summary.md    versioned, deterministic counts
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from support_copilot.data.source import MAX_CHARS, QUEUES, TECHNICAL_QUEUES, ensure_raw

logger = logging.getLogger(__name__)

N_FOLDS = 5  # one fold (~20%) becomes the test set
SEED = 42
PARAPHRASE_THRESHOLD = 0.7
OUT_DIR = Path("data/processed")
SUMMARY = Path("reports/prepare_summary.md")


def build_text(df: pd.DataFrame) -> pd.Series:
    """Model input = subject + body. The agent answer and tags are never features:
    they do not exist at inference time (see docs/dataset.md, leakage probe)."""
    subject = df["subject"].fillna("").astype(str).str.strip()
    body = df["body"].fillna("").astype(str).str.strip()
    return (subject + "\n" + body).str.strip()


def text_id(text: str) -> str:
    """Stable, non-reversible row identifier (no raw text in logs)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clean(df: pd.DataFrame, language: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter one language and apply the same rules the API contract implies."""
    stats: dict[str, int] = {}
    out = df[df["language"] == language].copy()
    stats["rows_language"] = len(out)

    out["text"] = build_text(out)
    empty = out["text"].str.len() == 0
    stats["dropped_empty"] = int(empty.sum())
    out = out[~empty]

    dup = out["text"].duplicated()
    stats["dropped_duplicates"] = int(dup.sum())
    out = out[~dup]

    too_long = out["text"].str.len() > MAX_CHARS
    stats["dropped_over_max_chars"] = int(too_long.sum())
    out = out[~too_long]

    out = out.rename(columns={"queue": "label"})
    out["is_technical"] = out["label"].isin(TECHNICAL_QUEUES)
    out["text_id"] = out["text"].map(text_id)
    stats["rows_clean"] = len(out)
    return out[["text_id", "text", "label", "is_technical"]].reset_index(drop=True), stats


def paraphrase_edges(texts: pd.Series, threshold: float, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray, int]:
    """Pairs (i<j) whose TF-IDF cosine >= threshold. Unsupervised: labels are never used."""
    x = TfidfVectorizer(sublinear_tf=True).fit_transform(texts)
    n = x.shape[0]
    rows, cols = [], []
    for start in range(0, n, chunk):
        sim = (x[start : start + chunk] @ x.T).tocoo()
        keep = (sim.data >= threshold) & ((sim.row + start) < sim.col)
        rows.append(sim.row[keep] + start)
        cols.append(sim.col[keep])
    return np.concatenate(rows), np.concatenate(cols), n


def paraphrase_groups(texts: pd.Series, threshold: float = PARAPHRASE_THRESHOLD) -> np.ndarray:
    """Connected components of the near-duplicate graph -> one group id per text."""
    r, c, n = paraphrase_edges(texts, threshold)
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(int)


def check_labels(df: pd.DataFrame) -> None:
    found = set(df["label"].unique())
    expected = set(QUEUES)
    if found != expected:
        raise ValueError(f"label set changed. missing={expected - found}, unexpected={found - expected}")


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified by queue, grouped by paraphrase family: no family on both sides."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(df["text"], df["label"], groups=df["group_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def cross_leakage(train: pd.DataFrame, test: pd.DataFrame, threshold: float = PARAPHRASE_THRESHOLD) -> float:
    """Share of test tickets with a train neighbour at cosine >= threshold (0 by construction)."""
    both = pd.concat([train["text"], test["text"]], ignore_index=True)
    r, c, _ = paraphrase_edges(both, threshold)
    n_train = len(train)
    crossing = ((r < n_train) & (c >= n_train)) | ((c < n_train) & (r >= n_train))
    test_hits = np.unique(np.where(r >= n_train, r, c)[crossing])
    return len(test_hits) / len(test)


def summarize(
    stats_en: dict[str, int],
    stats_de: dict[str, int],
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_sizes: np.ndarray,
    leakage: float,
) -> str:
    lines = [
        "# Prepare summary",
        "",
        f"Generated by `make prepare` · seed {SEED} · StratifiedGroupKFold({N_FOLDS}) first fold as test · "
        f"max chars {MAX_CHARS} · paraphrase threshold cos >= {PARAPHRASE_THRESHOLD}",
        "",
        "| step | en | de (drift pool) |",
        "|---|---|---|",
    ]
    for key in stats_en:
        lines.append(f"| {key} | {stats_en[key]:,} | {stats_de.get(key, 0):,} |")
    lines += [
        "",
        "## Paraphrase families (en)",
        "",
        f"- Families: {len(group_sizes):,} · singletons: {(group_sizes == 1).sum():,} · "
        f"multi-ticket: {(group_sizes > 1).sum():,} · largest: {group_sizes.max()}",
        f"- Families on both sides of the split: {len(set(train['group_id']) & set(test['group_id']))}",
        f"- Test tickets with a train neighbour at cos >= {PARAPHRASE_THRESHOLD}: {leakage:.1%} "
        "(was 57.4% with the plain random split)",
        "",
        "| queue | technical | train | test |",
        "|---|---|---|---|",
    ]
    tr, te = train["label"].value_counts(), test["label"].value_counts()
    for q in QUEUES:
        lines.append(f"| {q} | {'yes' if q in TECHNICAL_QUEUES else ''} | {tr.get(q, 0):,} | {te.get(q, 0):,} |")
    lines += [
        "",
        f"- Rows: train {len(train):,} · test {len(test):,} ({len(test) / (len(train) + len(test)):.1%})",
        f"- Technical share: train {train['is_technical'].mean():.1%} · test {test['is_technical'].mean():.1%}",
        f"- Train/test text_id overlap: {len(set(train['text_id']) & set(test['text_id']))}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    from support_copilot.data.schemas import validate_processed  # pandera import kept off the hot path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = pd.read_csv(ensure_raw())

    en, stats_en = clean(raw, "en")
    check_labels(en)
    en["group_id"] = paraphrase_groups(en["text"])
    group_sizes = np.bincount(en["group_id"])
    train, test = split(en)
    leakage = cross_leakage(train, test)
    de, stats_de = clean(raw, "de")
    de["group_id"] = -1  # not used for splitting; kept for a uniform schema

    for name, frame in (("train", train), ("test", test), ("drift_pool_de", de)):
        validate_processed(frame)
        logger.info("%s: %d rows, schema ok", name, len(frame))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    test.to_parquet(OUT_DIR / "test.parquet", index=False)
    de.to_parquet(OUT_DIR / "drift_pool_de.parquet", index=False)
    logger.info("train=%d test=%d drift_pool_de=%d leakage=%.4f", len(train), len(test), len(de), leakage)

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(summarize(stats_en, stats_de, train, test, group_sizes, leakage), encoding="utf-8", newline="\n")
    logger.info("summary written to %s", SUMMARY)


if __name__ == "__main__":
    main()
