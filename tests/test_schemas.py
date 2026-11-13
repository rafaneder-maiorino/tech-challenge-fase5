import pandas as pd
import pandera.errors
import pytest

from support_copilot.data.prepare import text_id
from support_copilot.data.schemas import validate_processed
from support_copilot.data.source import MAX_CHARS


def _row(text: str = "router is down", label: str = "Technical Support", is_technical: bool = True) -> dict:
    return {"text_id": text_id(text), "text": text, "label": label, "is_technical": is_technical, "group_id": 0}


def test_valid_batch_passes():
    df = pd.DataFrame([_row(), _row("refund please", "Billing and Payments", False)])
    assert len(validate_processed(df)) == 2


@pytest.mark.parametrize(
    "row",
    [
        _row(text="x" * (MAX_CHARS + 1)),
        _row(label="Brand New Queue"),
        _row(is_technical=False),  # contradicts the routing map
        {**_row(), "text": " padded "},
        {**_row(), "group_id": -5},
    ],
    ids=["too-long", "unknown-queue", "routing-inconsistent", "not-trimmed", "bad-group"],
)
def test_contract_violations_are_reported(row):
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_processed(pd.DataFrame([row]))


def test_duplicate_ids_rejected():
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_processed(pd.DataFrame([_row(), _row()]))


def test_unexpected_column_rejected():
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_processed(pd.DataFrame([{**_row(), "answer": "agent reply"}]))
