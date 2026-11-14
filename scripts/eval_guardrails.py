"""Measure the guardrails: block rate on attacks, false positives on legitimate questions.

Sets (see data/eval/guardrails_suite.yaml for the protocol):
  - attacks.dev / attacks.holdout   our suite; holdout was never used for tuning the v1 rules
  - legit: near-miss questions + the 34 retrieval questions (in and out of the manual)
  - external: deepset/prompt-injections (662 labelled prompts, en/de, Apache-2.0), pinned revision,
    a truly out-of-distribution test nobody here wrote

Layers:
  - input layer: rules v1 (data/eval/guardrails_rules_v1.yaml) vs current rules
  - end to end (current rules, real retriever, fake LLM): where each question stops
  - context layer: how many chunks of the real manual the indirect-injection rules flag

    uv run python scripts/eval_guardrails.py   # writes reports/guardrails_eval.md
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from support_copilot.guardrails import InputGuard, RuleSet
from support_copilot.rag.chat import ChatService, GuardrailBlocked
from support_copilot.rag.config import RagSettings
from support_copilot.rag.ingest import load_or_build
from support_copilot.rag.llm import FakeGenerator
from support_copilot.rag.retriever import ManualRetriever

logger = logging.getLogger("eval_guardrails")

SUITE = Path("data/eval/guardrails_suite.yaml")
RAG_QUESTIONS = Path("data/eval/rag_questions.yaml")
RULES_V1 = Path("data/eval/guardrails_rules_v1.yaml")
OUT = Path("reports/guardrails_eval.md")

EXTERNAL_DIR = Path("data/raw/external")
EXTERNAL_REV = "4f61ecb038e9c3fb77e21034b22511b523772cdd"
EXTERNAL_FILES = {
    "train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
    "test": "data/test-00000-of-00001-701d16158af87368.parquet",
}
EXTERNAL_SHA256 = {
    "train": "2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04",
    "test": "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e",
}
GERMAN = re.compile(r"\b(und|ich|nicht|ist|der|die|das|wie|ein|eine|mit|fur|sie|wir)\b", re.IGNORECASE)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = **{k / n:.1%}** (IC95% {lo:.0%}–{hi:.0%})" if n else "—"


def fetch_external() -> pd.DataFrame | None:
    frames = []
    for split, rel in EXTERNAL_FILES.items():
        path = EXTERNAL_DIR / Path(rel).name
        if not path.exists():
            url = f"https://huggingface.co/datasets/deepset/prompt-injections/resolve/{EXTERNAL_REV}/{rel}"
            try:
                EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed https URL, pinned revision
            except OSError as exc:
                logger.warning("external set unavailable (%s); skipping", exc)
                return None
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = EXTERNAL_SHA256.get(split)
        if expected and digest != expected:
            raise RuntimeError(f"{path} sha256 {digest} != pinned {expected}")
        logger.info("external %s sha256=%s", split, digest)
        df = pd.read_parquet(path)
        df["split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["lang"] = ["de" if len(GERMAN.findall(t)) >= 2 else "en" for t in out["text"]]
    return out


def input_layer(guard: InputGuard, suite: dict, rag: dict, external: pd.DataFrame | None) -> dict:
    def blocked(texts: list[str]) -> list[bool]:
        return [not guard.check(t).allowed for t in texts]

    dev = [a["text"] for a in suite["attacks"]["dev"]]
    hold = suite["attacks"]["holdout"]
    legit = suite["legit"]["near_miss"] + [q["q"] for q in rag["in_domain"]] + list(rag["out_of_domain"])
    res = {
        "dev": (sum(blocked(dev)), len(dev)),
        "holdout": (sum(blocked([a["text"] for a in hold])), len(hold)),
        "legit": (sum(blocked(legit)), len(legit)),
        "holdout_by_kind": Counter(),
        "holdout_total_by_kind": Counter(a["kind"] for a in hold),
        "holdout_missed": [a for a in hold if guard.check(a["text"]).allowed],
    }
    for a in hold:
        if not guard.check(a["text"]).allowed:
            res["holdout_by_kind"][a["kind"]] += 1
    if external is not None:
        flags = blocked(list(external["text"]))
        ext = external.assign(blocked=flags)
        for lang in ("en", "de", "all"):
            sub = ext if lang == "all" else ext[ext["lang"] == lang]
            inj, ben = sub[sub["label"] == 1], sub[sub["label"] == 0]
            res[f"ext_{lang}"] = ((int(inj["blocked"].sum()), len(inj)), (int(ben["blocked"].sum()), len(ben)))
        res["ext_fp_examples"] = list(ext[(ext["label"] == 0) & ext["blocked"]]["text"].head(5))
    return res


def outcome(service: ChatService, text: str) -> tuple[str, bool]:
    """(where the question stopped, whether the LLM was called)."""
    try:
        r = service.ask(text)
    except GuardrailBlocked:
        return "blocked_input", False
    return ("answered" if r.grounded else f"refused:{r.refusal_reason}"), r.llm_called


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    suite = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    rag = yaml.safe_load(RAG_QUESTIONS.read_text(encoding="utf-8"))
    external = fetch_external()
    current, v1 = RuleSet.load(), RuleSet.load(RULES_V1)
    layers = {"v1": input_layer(InputGuard(v1), suite, rag, external)}
    layers["current"] = input_layer(InputGuard(current), suite, rag, external)

    settings = RagSettings()
    store = load_or_build(settings)
    service = ChatService(ManualRetriever(store, settings.top_k, settings.min_score), FakeGenerator(), current)
    attacks = [a["text"] for a in suite["attacks"]["dev"] + suite["attacks"]["holdout"]]
    attack_runs = [outcome(service, t) for t in attacks]
    e2e_attacks = Counter(label for label, _ in attack_runs)
    reached = sum(called for _, called in attack_runs)
    e2e_in = Counter(outcome(service, q["q"])[0] for q in rag["in_domain"])
    e2e_near = Counter(outcome(service, q)[0] for q in suite["legit"]["near_miss"])
    e2e_out = Counter(outcome(service, q)[0] for q in rag["out_of_domain"])
    e2e_ext: dict[str, tuple[Counter, int]] = {}
    if external is not None:
        for label, name in ((1, "injections"), (0, "benign")):
            runs = [outcome(service, t) for t in external.loc[external["label"] == label, "text"]]
            e2e_ext[name] = (Counter(lbl for lbl, _ in runs), sum(called for _, called in runs))

    chunks = [str(d.metadata.get("body", d.page_content)) for d in store.docstore._dict.values()]  # noqa: SLF001
    flagged = Counter(m.rule for m in (current.first_match(c, "context") for c in chunks) if m)

    lines = [
        "# Guardrails evaluation",
        "",
        "Generated by `scripts/eval_guardrails.py`. Protocol in `data/eval/guardrails_suite.yaml`: the holdout",
        "attacks were written together with the dev attacks, before any rule existed, and were not used to",
        "tune rules **v1**. After the v1 holdout run, the misses were analysed and fixed (current rules), so",
        "for the current rules the holdout is no longer unseen. The external set (deepset/prompt-injections,",
        f"revision `{EXTERNAL_REV[:7]}`) was never used for tuning by either version.",
        "",
        "## Input layer: block rate and false positives",
        "",
        "| set | rules v1 | current rules |",
        "|---|---|---|",
    ]
    for key, label in [("dev", "attacks · dev (blocked)"), ("holdout", "attacks · holdout (blocked)")]:
        lines.append(f"| {label} | {rate(*layers['v1'][key])} | {rate(*layers['current'][key])} |")
    lines.append(
        f"| legit · near-miss + 34 retrieval questions (blocked = false positive) | {rate(*layers['v1']['legit'])} "
        f"| {rate(*layers['current']['legit'])} |"
    )
    if external is not None:
        for lang in ("en", "de", "all"):
            (k1, n1), (f1, m1) = layers["v1"][f"ext_{lang}"]
            (k2, n2), (f2, m2) = layers["current"][f"ext_{lang}"]
            lines.append(f"| external · {lang} · injections (blocked) | {rate(k1, n1)} | {rate(k2, n2)} |")
            lines.append(f"| external · {lang} · benign (blocked = false positive) | {rate(f1, m1)} | {rate(f2, m2)} |")
    lines += ["", "**Holdout by attack kind (v1 → current):**", "", "| kind | v1 | current |", "|---|---|---|"]
    for kind, total in sorted(layers["v1"]["holdout_total_by_kind"].items()):
        v1k, curk = layers["v1"]["holdout_by_kind"][kind], layers["current"]["holdout_by_kind"][kind]
        lines.append(f"| {kind} | {v1k}/{total} | {curk}/{total} |")
    lines += ["", "**Holdout attacks still passing the input layer (current rules):**", ""]
    lines += [f"- `{a['kind']}` — {a['text']}" for a in layers["current"]["holdout_missed"]] or ["- none"]
    if external is not None and layers["current"].get("ext_fp_examples"):
        lines += ["", "**External benign prompts blocked (first 5):**", ""]
        lines += [f"- {t[:160]}" for t in layers["current"]["ext_fp_examples"]]

    def fmt(c: Counter) -> str:
        return ", ".join(f"{k}: {v}" for k, v in sorted(c.items()))

    lines += [
        "",
        "## End to end (current rules, real retriever, fake LLM)",
        "",
        "Where each question stops. `answered` = the LLM was called and its answer passed the output layer.",
        "",
        "| set | n | outcome |",
        "|---|---|---|",
        f"| attacks (dev + holdout) | {sum(e2e_attacks.values())} | {fmt(e2e_attacks)} |",
        f"| legit · in the manual | {sum(e2e_in.values())} | {fmt(e2e_in)} |",
        f"| legit · near-miss | {sum(e2e_near.values())} | {fmt(e2e_near)} |",
        f"| legit · outside the manual | {sum(e2e_out.values())} | {fmt(e2e_out)} |",
        *(
            f"| external · {name} | {sum(c.values())} | {fmt(c)} · **LLM called: {n}** |"
            for name, (c, n) in e2e_ext.items()
        ),
        "",
        f"Attacks that reached the LLM: **{reached}/{sum(e2e_attacks.values())}**. The fake LLM is extractive,",
        "so for these the answer is a manual passage with its page; a real LLM still has the grounding",
        "prompt and the output layer (citations to retrieved pages) behind it.",
        "",
        "## Context layer: false positives on the real manual",
        "",
        f"Chunks flagged by the indirect-injection rules: **{sum(flagged.values())}/{len(chunks)}**"
        + (f" ({fmt(flagged)})" if flagged else "")
        + ". Every flagged chunk would be dropped before reaching the LLM.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
