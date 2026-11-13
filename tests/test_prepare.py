import pandas as pd
import pytest

from support_copilot.data.prepare import MAX_CHARS, build_text, check_labels, clean, split
from support_copilot.data.source import QUEUES, TECHNICAL_QUEUES


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


def test_split_is_stratified_and_disjoint():
    rows = [{"body": f"{q} ticket {i}", "queue": q} for q in QUEUES for i in range(20)]
    out, _ = clean(_raw(rows), "en")
    train, test = split(out)
    assert not set(train["text_id"]) & set(test["text_id"])
    assert (test["label"].value_counts() == 4).all()  # 20% of 20 per class
