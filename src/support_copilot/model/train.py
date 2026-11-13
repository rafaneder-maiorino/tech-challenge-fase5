"""Train the ticket-triage candidates, compare them with paired CV and register the champion.

Selection protocol (test set is touched only for final reporting):
1. Every candidate is scored with the SAME stratified 5-fold split of the training set, so the
   per-fold scores are paired and the champion-vs-runner-up delta has its own noise estimate.
2. The champion is the non-baseline candidate with the highest mean CV macro F1.
3. Every candidate is refit on the full training set and evaluated once on the test set.
4. One MLflow run per candidate (params, CV and test metrics, confusion matrix, report, model).
5. The champion is registered as ``ticket-classifier`` with alias ``champion`` and exported to
   ``models/ticket_classifier.joblib`` (+ ``.json`` metadata) for serving without MLflow.

Usage:
    uv run python -m support_copilot.model.train            # needs the MLflow server on :5001
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from support_copilot.data.source import QUEUES, TECHNICAL_QUEUES  # noqa: E402

logger = logging.getLogger(__name__)

SEED = 42
N_SPLITS = 5
EXPERIMENT = "ticket-triage"
REGISTERED_MODEL = "ticket-classifier"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

LABELS: np.ndarray = np.array(QUEUES)
TECH_MASK: np.ndarray = np.isin(LABELS, sorted(TECHNICAL_QUEUES))


def encode(labels: pd.Series) -> np.ndarray:
    index = {q: i for i, q in enumerate(QUEUES)}
    return labels.map(index).to_numpy(dtype=int)


def routing_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 of the technical-vs-not decision implied by the predicted queue."""
    return float(f1_score(TECH_MASK[y_true], TECH_MASK[y_pred], average="macro"))


def _tfidf(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=max_features)


@dataclass(frozen=True)
class Candidate:
    name: str
    build: Callable[[], Pipeline]
    params: dict[str, Any]
    balanced_sample_weight: bool = False  # passed at fit time (XGBoost has no class_weight)
    baseline: bool = False
    # Non-sklearn classes MLflow's skops serializer must be told to trust (explicit allow-list).
    trusted_types: tuple[str, ...] = ()


def default_candidates(xgb_estimators: int = 150) -> list[Candidate]:
    return [
        Candidate(
            "majority",
            lambda: Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
            {"strategy": "most_frequent"},
            baseline=True,
        ),
        Candidate(
            "tfidf_logreg",
            lambda: Pipeline(
                [
                    ("tfidf", _tfidf(100_000)),
                    ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)),
                ]
            ),
            {"ngram_range": "1-2", "max_features": 100_000, "C": 4.0, "class_weight": "balanced"},
        ),
        Candidate(
            "tfidf_linearsvc",
            lambda: Pipeline(
                [
                    ("tfidf", _tfidf(100_000)),
                    (
                        "clf",
                        CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid"),
                    ),
                ]
            ),
            {"ngram_range": "1-2", "max_features": 100_000, "C": 0.5, "calibration": "sigmoid-cv3"},
            trusted_types=("sklearn.calibration._CalibratedClassifier", "sklearn.calibration._SigmoidCalibration"),
        ),
        Candidate(
            "tfidf_xgboost",
            lambda: Pipeline(
                [
                    ("tfidf", _tfidf(20_000)),
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=xgb_estimators,
                            max_depth=6,
                            learning_rate=0.2,
                            subsample=0.8,
                            colsample_bytree=0.5,
                            tree_method="hist",
                            eval_metric="mlogloss",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "ngram_range": "1-2",
                "max_features": 20_000,
                "n_estimators": xgb_estimators,
                "max_depth": 6,
                "learning_rate": 0.2,
                "colsample_bytree": 0.5,
                "sample_weight": "balanced",
            },
            balanced_sample_weight=True,
            trusted_types=("xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"),
        ),
    ]


@dataclass
class Result:
    candidate: Candidate
    cv: dict[str, np.ndarray]
    test: dict[str, float] = field(default_factory=dict)
    run_id: str = ""
    pipeline: Pipeline | None = None
    y_pred: np.ndarray | None = None


def _fit_params(candidate: Candidate, y: np.ndarray) -> dict[str, Any]:
    if candidate.balanced_sample_weight:
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def cross_validate_candidate(
    candidate: Candidate, x: pd.Series, y: np.ndarray, groups: np.ndarray, n_splits: int
) -> dict[str, np.ndarray]:
    """Grouped CV: paraphrase families never straddle train/validation folds (see prepare.py)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scoring = {
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
        "routing_f1": make_scorer(routing_f1),
    }
    out = cross_validate(
        candidate.build(), x, y, groups=groups, cv=cv, scoring=scoring, params=_fit_params(candidate, y)
    )
    return {k.removeprefix("test_"): v for k, v in out.items() if k.startswith("test_")}


def confusion_png(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(QUEUES)), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ticks = [f"{q}{' *' if TECH_MASK[i] else ''}" for i, q in enumerate(QUEUES)]
    ax.set_xticks(range(len(QUEUES)), ticks, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(QUEUES)), ticks, fontsize=8)
    for i in range(len(QUEUES)):
        for j in range(len(QUEUES)):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > 0.5 else "black",
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (row-normalized)")
    ax.set_title(f"{title}  (* = routed to RAG)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(fig)


def error_structure(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Where do the errors live? Within the technical group, within non-technical, or across."""
    wrong = y_true != y_pred
    t_true, t_pred = TECH_MASK[y_true], TECH_MASK[y_pred]
    n = int(wrong.sum())
    within_tech = int((wrong & t_true & t_pred).sum())
    within_non = int((wrong & ~t_true & ~t_pred).sum())
    across = n - within_tech - within_non
    return {
        "errors": n,
        "within_technical": within_tech,
        "within_non_technical": within_non,
        "across_routing_boundary": across,
        "share_within_technical": within_tech / n if n else 0.0,
        "share_across_boundary": across / n if n else 0.0,
    }


def run(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tracking_uri: str,
    candidates: list[Candidate],
    n_splits: int = N_SPLITS,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    x_train, y_train = train["text"], encode(train["label"])
    groups = train["group_id"].to_numpy()
    x_test, y_test = test["text"], encode(test["label"])

    results: list[Result] = []
    for cand in candidates:
        logger.info("cv %s", cand.name)
        t0 = time.perf_counter()
        cv_scores = cross_validate_candidate(cand, x_train, y_train, groups, n_splits)
        cv_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        pipe = cand.build()
        pipe.fit(x_train, y_train, **_fit_params(cand, y_train))
        fit_seconds = time.perf_counter() - t0
        y_pred = pipe.predict(x_test)

        res = Result(cand, cv_scores, pipeline=pipe, y_pred=y_pred)
        res.test = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "test_f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "test_routing_f1": routing_f1(y_test, y_pred),
        }

        with mlflow.start_run(run_name=cand.name) as active:
            res.run_id = active.info.run_id
            mlflow.set_tags(
                {"candidate": cand.name, "baseline": str(cand.baseline).lower(), "split": "grouped-paraphrase"}
            )
            mlflow.log_params({"model": cand.name, "seed": SEED, "cv_splits": n_splits, **cand.params})
            mlflow.log_params({"n_train": len(train), "n_test": len(test), "n_classes": len(QUEUES)})
            for metric, values in cv_scores.items():
                mlflow.log_metric(f"cv_{metric}_mean", float(np.mean(values)))
                mlflow.log_metric(f"cv_{metric}_std", float(np.std(values)))
            mlflow.log_metrics(res.test)
            mlflow.log_metrics({"cv_seconds": cv_seconds, "fit_seconds": fit_seconds})

            png = reports_dir / "_build" / f"confusion_{cand.name}.png"
            confusion_png(y_test, y_pred, png, f"{cand.name} — test")
            mlflow.log_artifact(str(png))
            report = classification_report(
                y_test, y_pred, labels=range(len(QUEUES)), target_names=QUEUES, output_dict=True, zero_division=0
            )
            mlflow.log_dict(report, "classification_report.json")

            sample = x_test.head(5).tolist()
            signature = infer_signature(sample, pipe.predict_proba(sample))
            mlflow.sklearn.log_model(
                pipe,
                name="model",
                signature=signature,
                skops_trusted_types=list(cand.trusted_types) or None,
            )
        logger.info(
            "%s cv_f1=%.3f±%.3f test_f1=%.3f routing=%.3f",
            cand.name,
            np.mean(cv_scores["f1_macro"]),
            np.std(cv_scores["f1_macro"]),
            res.test["test_f1_macro"],
            res.test["test_routing_f1"],
        )
        results.append(res)

    ranked = sorted(
        (r for r in results if not r.candidate.baseline),
        key=lambda r: float(np.mean(r.cv["f1_macro"])),
        reverse=True,
    )
    champion, runner_up = ranked[0], ranked[1] if len(ranked) > 1 else None

    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{champion.run_id}/model", REGISTERED_MODEL).version
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version)
    client.set_tag(champion.run_id, "champion", "true")

    errors = error_structure(y_test, champion.y_pred)
    paired = None
    if runner_up is not None:
        delta = champion.cv["f1_macro"] - runner_up.cv["f1_macro"]
        paired = {
            "runner_up": runner_up.candidate.name,
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "wins": int((delta > 0).sum()),
            "folds": int(delta.size),
        }

    meta = {
        "registered_model": REGISTERED_MODEL,
        "version": int(version),
        "model_version": f"{REGISTERED_MODEL}/v{version}",
        "run_id": champion.run_id,
        "candidate": champion.candidate.name,
        "labels": list(QUEUES),
        "technical_queues": sorted(TECHNICAL_QUEUES),
        "metrics": {**champion.test, "cv_f1_macro_mean": float(np.mean(champion.cv["f1_macro"]))},
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion.pipeline, models_dir / "ticket_classifier.joblib", compress=3)
    (models_dir / "ticket_classifier.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    confusion_png(
        y_test, champion.y_pred, reports_dir / "confusion_matrix.png", f"champion: {champion.candidate.name} — test"
    )
    (reports_dir / "model_selection.md").write_text(
        render_report(results, champion, paired, errors, y_test, n_splits), encoding="utf-8", newline="\n"
    )
    logger.info("champion %s registered as %s v%s", champion.candidate.name, REGISTERED_MODEL, version)
    return {"meta": meta, "paired": paired, "errors": errors}


def render_report(
    results: list[Result],
    champion: Result,
    paired: dict[str, Any] | None,
    errors: dict[str, float],
    y_test: np.ndarray,
    n_splits: int,
) -> str:
    lines = [
        "# Model selection — ticket triage",
        "",
        f"Generated by `python -m support_copilot.model.train` · seed {SEED} · "
        f"paired stratified-grouped {n_splits}-fold CV on train (paraphrase families never straddle folds), "
        "single evaluation on a group-disjoint test set.",
        "",
        "| candidate | CV macro F1 | test accuracy | test macro F1 | test weighted F1 | routing F1 (test) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        mark = " **(champion)**" if r is champion else ""
        cv = r.cv["f1_macro"]
        lines.append(
            f"| {r.candidate.name}{mark} | {cv.mean():.3f} ± {cv.std():.3f} | {r.test['test_accuracy']:.3f} | "
            f"{r.test['test_f1_macro']:.3f} | {r.test['test_f1_weighted']:.3f} | {r.test['test_routing_f1']:.3f} |"
        )
    lines += [
        "",
        "Champion = non-baseline candidate with the highest mean CV macro F1 (the test set does not pick it).",
        "",
    ]
    if paired:
        lines += [
            "## Champion vs runner-up (paired by fold)",
            "",
            f"- Runner-up: `{paired['runner_up']}`",
            f"- Delta CV macro F1: {paired['delta_mean']:+.4f} ± {paired['delta_std']:.4f} "
            f"· champion wins {paired['wins']}/{paired['folds']} folds",
            "",
        ]
    lines += [
        "## Where the champion's errors live (test)",
        "",
        "| bucket | errors | share |",
        "|---|---|---|",
        f"| within technical queues | {errors['within_technical']} | {errors['share_within_technical']:.1%} |",
        f"| within non-technical queues | {errors['within_non_technical']} | "
        f"{errors['within_non_technical'] / max(errors['errors'], 1):.1%} |",
        f"| across the routing boundary | {errors['across_routing_boundary']} | "
        f"{errors['share_across_boundary']:.1%} |",
        f"| **total** | {errors['errors']} | of {len(y_test)} test tickets |",
        "",
        "Only errors across the routing boundary change whether a ticket reaches the RAG.",
        "",
        "## Champion per-queue F1 (test)",
        "",
        "| queue | technical | F1 | support |",
        "|---|---|---|---|",
    ]
    report = classification_report(
        y_test, champion.y_pred, labels=range(len(QUEUES)), target_names=QUEUES, output_dict=True, zero_division=0
    )
    for i, q in enumerate(QUEUES):
        lines.append(
            f"| {q} | {'yes' if TECH_MASK[i] else ''} | {report[q]['f1-score']:.3f} | {int(report[q]['support'])} |"
        )
    lines += ["", "![confusion matrix](confusion_matrix.png)", ""]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    try:
        train = pd.read_parquet(PROCESSED / "train.parquet")
        test = pd.read_parquet(PROCESSED / "test.parquet")
    except FileNotFoundError:
        logger.error("data/processed not found — run `make prepare` first")
        return 1
    try:
        run(train, test, tracking_uri, default_candidates())
    except MlflowException as exc:
        if "Connection" in str(exc) or "Max retries" in str(exc):
            logger.error("MLflow server not reachable at %s — start it with `make mlflow-server`", tracking_uri)
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
