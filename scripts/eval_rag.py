"""RAG evaluation: cosine threshold (D5) vs cross-encoder reranker (D7), on data never used to tune.

Protocol (fixed before the reranker touched the golden set; see git history of this file):
- Dev set  data/eval/rag_questions.yaml: chose chunk size and the cosine threshold (D4/D5). Here it
  also sets the reranker threshold: midpoint between the lowest in-domain and the highest
  out-of-domain best logit — the same rule used for the cosine threshold.
- Golden   data/eval/rag_golden.yaml: pre-registered (commit c38094b), evaluated once per config.
- Tickets  data/processed/test.parquet through /ticket: the dataset's products are not OPNsense, so
  every "answered" ticket is a candidate false answer (a 30-ticket review of the cosine run found
  0 relevant; see docs/findings.md §7).
- Adopt the reranker only if, vs cosine: golden in-domain answered >=, hard negatives refused >,
  and tickets answered <. Otherwise keep cosine and report.

Writes reports/rag_eval.md (+ MLflow run per config when the server is up).
Run: uv run python scripts/eval_rag.py   (~15 min: the reranker reads ~2,100 tickets x 10 passages)
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from support_copilot.api.classifier import TicketClassifier
from support_copilot.rag.chat import ChatService, GuardrailBlocked
from support_copilot.rag.config import EMBEDDING_MODEL, INGEST_VERSION, RERANKER_MODEL, RagSettings
from support_copilot.rag.ingest import load_or_build
from support_copilot.rag.llm import build_generator
from support_copilot.rag.retriever import Hit, ManualRetriever, load_reranker
from support_copilot.ticket import TicketService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_rag")

DEV = Path("data/eval/rag_questions.yaml")
GOLDEN = Path("data/eval/rag_golden.yaml")
OUT = Path("reports/rag_eval.md")
EXPERIMENT = "rag-evaluation"
SAMPLE = 10  # answered tickets shown in the report for manual audit


class Recording(ManualRetriever):
    """Retriever that remembers the best evidence score of its last call (also when nothing passes)."""

    last_best: float = float("nan")

    def ranked(self, q: str) -> list[Hit]:
        hits = self.search(q, k=self.fetch_k if self.reranker else self.top_k)
        return self.rerank(q, hits)

    def relevant(self, question: str) -> list[Hit]:
        ranked = self.ranked(question)
        self.last_best = best_score(self, ranked)
        thr = self.rerank_min if self.reranker else self.min_score
        return [h for h in ranked[: self.top_k] if score_of(self, h) >= thr]


def score_of(r: ManualRetriever, h: Hit) -> float:
    return float(h.rerank) if r.reranker is not None else h.score


def best_score(r: ManualRetriever, ranked: list[Hit]) -> float:
    return score_of(r, ranked[0]) if ranked else float("nan")


def golden_retrieval(r: Recording, items: list[dict]) -> tuple[dict, list[dict], np.ndarray]:
    rows, best = [], []
    for it in items:
        ok = {it["section"], *it.get("accept", [])}
        ranked = r.ranked(it["q"])
        rank = next((i + 1 for i, h in enumerate(ranked) if h.section in ok), None)
        best.append(best_score(r, ranked))
        rows.append(
            {"q": it["q"], "expected": it["section"], "top1": ranked[0].section, "rank": rank, "best": best[-1]}
        )
    ranks = [x["rank"] for x in rows]
    m = {
        "hit@1": float(np.mean([k == 1 for k in ranks])),
        "hit@4": float(np.mean([k is not None and k <= 4 for k in ranks])),
        "mrr": float(np.mean([1 / k if k else 0.0 for k in ranks])),
    }
    return m, rows, np.array(best)


def best_of(r: Recording, qs: list[str]) -> np.ndarray:
    return np.array([best_score(r, r.ranked(q)) for q in qs])


def calibrate(r: Recording, dev: dict) -> tuple[float, float, float]:
    sc_in = best_of(r, [it["q"] for it in dev["in_domain"]])
    sc_out = best_of(r, dev["out_of_domain"])
    return float((sc_in.min() + sc_out.max()) / 2), float(sc_in.min()), float(sc_out.max())


def outcome(service: ChatService, q: str) -> tuple[str, list[Hit]]:
    try:
        res = service.ask(q)
    except GuardrailBlocked as b:
        return f"blocked:{b.verdict.guardrail}", []
    return ("answered" if res.grounded else f"refused:{res.refusal_reason}"), res.sources


def run_config(name: str, r: Recording, golden: dict, test: pd.DataFrame, clf: TicketClassifier) -> dict:
    thr = r.rerank_min if r.reranker else r.min_score
    service = ChatService(r, build_generator())
    ind, easy, hard = golden["in_domain"], golden["out_of_domain"]["easy"], golden["out_of_domain"]["hard"]
    m, rows, sc_in = golden_retrieval(r, ind)
    sc_easy, sc_hard = best_of(r, easy), best_of(r, hard)
    e2e_in = [outcome(service, it["q"]) for it in ind]
    e2e_out = {q: outcome(service, q)[0] for q in easy + hard}
    grounded = sum(o == "answered" for o, _ in e2e_in)
    cited_ok = sum(
        1
        for it, (o, src) in zip(ind, e2e_in, strict=True)
        if o == "answered" and src[0].section in {it["section"], *it.get("accept", [])}
    )

    tickets = TicketService(clf, service)
    t0 = time.perf_counter()
    recs = []
    for text in test.text:
        res = tickets.handle(text)
        recs.append(
            {
                "action": res.action,
                "reason": res.escalation_reason,
                "guard": f"{res.blocked_by.guardrail}/{res.blocked_by.category}" if res.blocked_by else None,
                "best": r.last_best if res.action != "routed" else np.nan,
                "section": res.chat.sources[0].section if res.chat and res.chat.sources else None,
                "excerpt": " ".join(text.split())[:120] if res.action == "answered" else None,
            }
        )
        r.last_best = float("nan")
    secs = time.perf_counter() - t0
    df = pd.DataFrame(recs)
    df["truth_tech"] = test.is_technical.to_numpy()
    tech = df[df.action != "routed"]
    log.info("%s: tickets done in %.0f s", name, secs)
    return {
        "name": name,
        "thr": thr,
        "m": m,
        "rows": rows,
        "sc_in": sc_in,
        "sc_easy": sc_easy,
        "sc_hard": sc_hard,
        "hard": hard,
        "e2e_out": e2e_out,
        "grounded": grounded,
        "cited_ok": cited_ok,
        "n_in": len(ind),
        "answer_in": float(np.mean(sc_in >= thr)),
        "refuse_easy": float(np.mean(sc_easy < thr)),
        "refuse_hard": float(np.mean(sc_hard < thr)),
        "df": df,
        "tech": tech,
        "ms": 1000 * secs / len(df),
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def section(res: dict) -> list[str]:
    m, df, tech, n = res["m"], res["df"], res["tech"], len(res["df"])
    fmt = ".2f" if res["name"] == "rerank" else ".3f"
    trio = lambda a: f"{a.min():{fmt}} · {np.median(a):{fmt}} · {a.max():{fmt}}"  # noqa: E731
    ans = df[df.action == "answered"]
    L = [
        f"### `{res['name']}` — threshold {res['thr']:{fmt}}",
        "",
        f"Retrieval on the golden set: hit@1 **{m['hit@1']:.3f}** · hit@4 **{m['hit@4']:.3f}** · "
        f"MRR **{m['mrr']:.3f}**",
        "",
        "| group | n | best score min · median · max | at the threshold |",
        "|---|---|---|---|",
        f"| in-domain | {res['n_in']} | {trio(res['sc_in'])} | answered {pct(res['answer_in'])} |",
        f"| out-of-domain, easy | {len(res['sc_easy'])} | {trio(res['sc_easy'])} | refused {pct(res['refuse_easy'])} |",
        f"| out-of-domain, hard | {len(res['sc_hard'])} | {trio(res['sc_hard'])} | refused {pct(res['refuse_hard'])} |",
        "",
        "| hard negative | best score | end to end |",
        "|---|---|---|",
    ]
    L += [f"| {q} | {sc:{fmt}} | {res['e2e_out'][q]} |" for q, sc in zip(res["hard"], res["sc_hard"], strict=True)]
    misses = [x for x in res["rows"] if x["rank"] != 1]
    L += [
        "",
        "Golden questions not at rank 1: "
        + (
            "none"
            if not misses
            else "; ".join(
                f'"{x["q"]}" → {x["top1"]} (expected {x["expected"]}, rank {x["rank"] or ">fetch"})' for x in misses
            )
        ),
    ]
    L += [
        "",
        f"End to end, in-domain: answered {res['grounded']}/{res['n_in']}, first source in the expected section "
        f"{res['cited_ok']}/{res['n_in']}. Out-of-domain: {dict(Counter(res['e2e_out'].values()))}.",
        "",
        f"`/ticket` over {n} test tickets: routed {pct(np.mean(df.action == 'routed'))} · **answered "
        f"{int((df.action == 'answered').sum())} ({pct(np.mean(df.action == 'answered'))})** · escalated "
        f"{pct(np.mean(df.action == 'escalated'))} · guardrail blocks on real tickets "
        f"{int(tech.guard.notna().sum())}/{len(tech)} · {res['ms']:.0f} ms per ticket",
        f"Best evidence score of all {len(tech)} technical tickets: median {np.nanmedian(tech.best):{fmt}} · "
        f"p90 {np.nanpercentile(tech.best, 90):{fmt}} · p99 {np.nanpercentile(tech.best, 99):{fmt}}",
        f"Answered sections: {dict(Counter(ans.section).most_common(5))}",
    ]
    if len(ans):
        L += ["", f"Random answered tickets (up to {SAMPLE}, for audit):", ""]
        L += [f"- `{a.section}` — {a.excerpt}…" for a in ans.sample(min(SAMPLE, len(ans)), random_state=7).itertuples()]
    return L


def log_mlflow(res: dict, params: dict) -> str:
    try:
        import mlflow
        import requests

        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
        requests.get(uri, timeout=2)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT)
        df = res["df"]
        metrics = {
            "golden_hit_at_1": res["m"]["hit@1"],
            "golden_hit_at_4": res["m"]["hit@4"],
            "golden_mrr": res["m"]["mrr"],
            "golden_answer_rate_in": res["answer_in"],
            "golden_refusal_easy": res["refuse_easy"],
            "golden_refusal_hard": res["refuse_hard"],
            "tickets_answered": float(np.mean(df.action == "answered")),
            "tickets_escalated": float(np.mean(df.action == "escalated")),
            "ms_per_ticket": res["ms"],
        }
        with mlflow.start_run(run_name=f"rag-{res['name']}") as run:
            mlflow.log_params({**params, "retrieval": res["name"], "threshold": res["thr"]})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(OUT))
        return f"`{run.info.run_id[:8]}`"
    except Exception as exc:  # the report is the source of truth; MLflow is a copy
        log.warning("MLflow not logged (%s): start it with `make mlflow-server`", type(exc).__name__)
        return "not logged (server down)"


def main() -> None:
    s = RagSettings()
    store = load_or_build(s)
    dev = yaml.safe_load(DEV.read_text(encoding="utf-8"))
    golden = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    clf = TicketClassifier.load(Path("models"))
    test = pd.read_parquet("data/processed/test.parquet")

    cos = Recording(store, s.top_k, s.min_score)
    rr = Recording(
        store, s.top_k, s.min_score, reranker=load_reranker(replace(s, reranker=RERANKER_MODEL)), fetch_k=s.fetch_k
    )
    rr.rerank_min, lo, hi = calibrate(rr, dev)
    log.info("reranker threshold from dev: %.3f (in-domain min %.2f, out-of-domain max %.2f)", rr.rerank_min, lo, hi)

    results = [run_config("cosine", cos, golden, test, clf), run_config("rerank", rr, golden, test, clf)]
    c, k = results
    ans = lambda x: int((x["df"].action == "answered").sum())  # noqa: E731
    checks = {
        "golden in-domain answered >= cosine": k["answer_in"] >= c["answer_in"],
        "hard negatives refused > cosine": k["refuse_hard"] > c["refuse_hard"],
        "tickets answered < cosine": ans(k) < ans(c),
    }
    adopt = all(checks.values())

    params = {
        "embedding": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "ingest_version": INGEST_VERSION,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "top_k": s.top_k,
        "fetch_k": s.fetch_k,
        "provider": build_generator().name,
        "classifier": clf.model_version,
    }
    L = [
        "# RAG evaluation — cosine threshold vs cross-encoder reranker",
        "",
        f"Generated by `scripts/eval_rag.py` · protocol in its docstring · golden set `{GOLDEN}` (pre-registered) · "
        f"chunk {s.chunk_size}/{s.chunk_overlap} · k={s.top_k} · reranker `{RERANKER_MODEL}` over fetch_k="
        f"{s.fetch_k} · provider `{params['provider']}`",
        "",
        f"Reranker threshold, from the dev set only: **{rr.rerank_min:.2f}** (midpoint of in-domain min {lo:.2f} "
        f"and out-of-domain max {hi:.2f}).",
        "",
        "## Summary",
        "",
        "| | cosine (D5) | rerank (D7) |",
        "|---|---|---|",
        f"| golden hit@1 / MRR | {c['m']['hit@1']:.3f} / {c['m']['mrr']:.3f} | "
        f"{k['m']['hit@1']:.3f} / {k['m']['mrr']:.3f} |",
        f"| golden in-domain answered | {pct(c['answer_in'])} | {pct(k['answer_in'])} |",
        f"| golden easy negatives refused | {pct(c['refuse_easy'])} | {pct(k['refuse_easy'])} |",
        f"| golden hard negatives refused | {pct(c['refuse_hard'])} | {pct(k['refuse_hard'])} |",
        f"| test tickets answered (candidate false answers) | {ans(c)} | {ans(k)} |",
        f"| latency per ticket | {c['ms']:.0f} ms | {k['ms']:.0f} ms |",
        "",
        "Adoption rule: "
        + " · ".join(f"{name}: {'yes' if ok else 'NO'}" for name, ok in checks.items())
        + f" → **{'adopt the reranker' if adopt else 'keep cosine'}**",
        "",
        "## Details",
        "",
    ]
    for res in results:
        L += section(res) + [""]
    L += [
        "## Tracking",
        "",
        "MLflow runs: (filled below)",
        "",
        "Faithfulness of generated text is not measured with the `fake` provider (its answers are copies of the "
        "retrieved passage). It needs a real LLM key: see README.",
    ]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    runs = " · ".join(f"{res['name']} {log_mlflow(res, params)}" for res in results)
    OUT.write_text(
        OUT.read_text(encoding="utf-8").replace("MLflow runs: (filled below)", f"MLflow runs: {runs}"), encoding="utf-8"
    )
    log.info("adopt reranker: %s %s", adopt, checks)
    log.info("written %s", OUT)


if __name__ == "__main__":
    main()
