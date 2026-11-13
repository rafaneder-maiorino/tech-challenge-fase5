import json

import joblib
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from support_copilot.data.source import QUEUES, TECHNICAL_QUEUES
from support_copilot.model import train as tr


def _frame(n_per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for q in QUEUES:
        vocab = [f"{q.split()[0].lower()}{k}" for k in range(5)] + ["please", "help", "issue"]
        for _ in range(n_per_class):
            rows.append({"text": " ".join(rng.choice(vocab, 12)), "label": q})
    df = pd.DataFrame(rows)
    df["is_technical"] = df["label"].isin(TECHNICAL_QUEUES)
    df["group_id"] = np.arange(len(df))
    return df


def test_routing_f1_only_counts_the_technical_boundary():
    tech = [i for i, q in enumerate(QUEUES) if q in TECHNICAL_QUEUES]
    non = [i for i, q in enumerate(QUEUES) if q not in TECHNICAL_QUEUES]
    y_true = np.array([tech[0], non[0]])
    # wrong queue but same side of the boundary -> routing is still perfect
    assert tr.routing_f1(y_true, np.array([tech[1], non[1]])) == 1.0
    assert tr.routing_f1(y_true, np.array([non[0], tech[0]])) == 0.0


def test_error_structure_buckets_add_up():
    tech = [i for i, q in enumerate(QUEUES) if q in TECHNICAL_QUEUES]
    non = [i for i, q in enumerate(QUEUES) if q not in TECHNICAL_QUEUES]
    y_true = np.array([tech[0], non[0], tech[0], non[0]])
    y_pred = np.array([tech[1], non[1], non[0], non[0]])
    e = tr.error_structure(y_true, y_pred)
    assert (e["errors"], e["within_technical"], e["within_non_technical"], e["across_routing_boundary"]) == (3, 1, 1, 1)


@pytest.mark.slow
def test_run_registers_champion_and_exports_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    candidates = [c for c in tr.default_candidates(xgb_estimators=5) if c.name != "tfidf_linearsvc"]
    out = tr.run(
        _frame(12, 0),
        _frame(4, 1),
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        candidates=candidates,
        n_splits=2,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    meta = json.loads((tmp_path / "models" / "ticket_classifier.json").read_text())
    assert meta["candidate"] != "majority"
    assert meta["labels"] == list(QUEUES)
    pipe = joblib.load(tmp_path / "models" / "ticket_classifier.joblib")
    assert pipe.predict_proba(["technical0 technical1"]).shape == (1, len(QUEUES))
    assert (tmp_path / "reports" / "model_selection.md").exists()
    alias = MlflowClient().get_model_version_by_alias(tr.REGISTERED_MODEL, "champion")
    assert int(alias.version) == meta["version"]
    assert out["paired"]["folds"] == 2
