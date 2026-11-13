"""Pandera contract for processed ticket batches (train/test/drift pool and, later, inference logs).

The API validates ONE ticket with pydantic; this validates a BATCH with the same rules, so the
data the model learns from and the data it is served obey the same contract.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from support_copilot.data.source import MAX_CHARS, QUEUES, TECHNICAL_QUEUES


def _routing_consistent(df: pd.DataFrame) -> pd.Series:
    return df["is_technical"] == df["label"].isin(TECHNICAL_QUEUES)


PROCESSED_SCHEMA = pa.DataFrameSchema(
    {
        "text_id": pa.Column(str, pa.Check.str_length(16, 16), unique=True),
        "text": pa.Column(str, [pa.Check.str_length(1, MAX_CHARS), pa.Check(lambda s: s.str.strip() == s)]),
        "label": pa.Column(str, pa.Check.isin(QUEUES)),
        "is_technical": pa.Column(bool),
        # paraphrase family (prepare.paraphrase_groups); -1 where grouping is not used (drift pool)
        "group_id": pa.Column(int, pa.Check.ge(-1)),
    },
    checks=[pa.Check(_routing_consistent, error="is_technical must follow the queue -> RAG mapping")],
    strict=True,
    coerce=False,
)


def validate_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Raise pandera.errors.SchemaErrors listing every violation (lazy) instead of the first one."""
    return PROCESSED_SCHEMA.validate(df, lazy=True)
