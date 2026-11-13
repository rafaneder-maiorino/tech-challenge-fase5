import pandas as pd
import pytest

from support_copilot.data.prepare import (
    build_text,
    check_labels,
    clean,
    cross_leakage,
    paraphrase_groups,
    split,
)
from support_copilot.data.source import MAX_CHARS, QUEUES, TECHNICAL_QUEUES


def _raw(rows: list[dict]) -> pd.DataFrame:
    base = {"subject": "s", "body": "b", "answer": "secret answer", "queue": "Technical Support", "language": "en"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_build_text_uses_subject_and_body_only():
    df = _raw([{"subject": None, "body": "  router down  "}])
    assert build_text(df).iloc[0] == "router down"
    assert "secret" not in build_text(df).iloc[0]


def test_clean_filters_language_empty_duplicates_and_length():
    df = _raw(
        [
            {"body": "a"},
            {"body": "a"},  # duplicate
            {"subject": "", "body": ""},  # empty
            {"body": "x" * (MAX_CHARS + 1)},  # too long
            {"body": "hallo", "language": "de"},
        ]
    )
    out, stats = clean(df, "en")
    assert len(out) == 1
    assert stats == {
        "rows_language": 4,
        "dropped_empty": 1,
        "dropped_duplicates": 1,
        "dropped_over_max_chars": 1,
        "rows_clean": 1,
    }
    assert list(out.columns) == ["text_id", "text", "label", "is_technical"]


def test_is_technical_follows_queue_mapping():
    df = _raw([{"body": f"t{i}", "queue": q} for i, q in enumerate(QUEUES)])
    out, _ = clean(df, "en")
    assert set(out.loc[out["is_technical"], "label"]) == TECHNICAL_QUEUES


def test_check_labels_fails_on_unexpected_queue():
    df = pd.DataFrame({"label": [*QUEUES, "Brand New Queue"]})
    with pytest.raises(ValueError, match="unexpected"):
        check_labels(df)


def test_paraphrase_groups_join_near_duplicates_only():
    texts = pd.Series(
        [
            "my vpn connection drops every few minutes after the update",
            "my vpn connection drops every few minutes after the latest update",  # paraphrase of 0
            "please refund the duplicate charge on my invoice",
            "the printer on floor three is jammed again",
        ]
    )
    g = paraphrase_groups(texts, threshold=0.7)
    assert g[0] == g[1]
    assert len({g[0], g[2], g[3]}) == 3


def _families() -> pd.DataFrame:
    """20 families per queue, 3 paraphrases each: a random split would leak families."""
    rows = []
    for q in QUEUES:
        word = q.split()[0].lower()
        for f in range(20):
            for p in range(3):
                rows.append({"body": f"{word} family{f} unique{f} token{f} variant {p}", "queue": q})
    out, _ = clean(_raw(rows), "en")
    out["group_id"] = paraphrase_groups(out["text"], threshold=0.7)
    return out


def test_split_keeps_families_on_one_side_and_is_stratified():
    df = _families()
    train, test = split(df)
    assert not set(train["group_id"]) & set(test["group_id"])
    assert not set(train["text_id"]) & set(test["text_id"])
    assert set(test["label"]) == set(QUEUES)
    assert cross_leakage(train, test) == 0.0
