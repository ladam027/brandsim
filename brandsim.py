"""brandsim: score how closely candidate text matches a set of reference texts.

Built as an eval-harness piece for brand-aligned LLM output: give it on-brand
reference answers, feed it model generations, get a per-item score, a
pass/fail against a threshold, and a pass rate you can gate CI on.

Score = weighted blend of
  semantic  cosine similarity of sentence embeddings (meaning, tone)
  rouge_l   longest-common-subsequence F1 over tokens (phrasing, structure)
  jaccard   token-set overlap (vocabulary, brand terms)
Each metric is taken against the *best-matching* reference, so one strong
match is enough; references don't have to agree with each other.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence

__version__ = "0.1.0"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_WEIGHTS = {"semantic": 0.6, "rouge_l": 0.25, "jaccard": 0.15}

Embedder = Callable[[Sequence[str]], List[List[float]]]

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def rouge_l(a: Sequence[str], b: Sequence[str]) -> float:
    """ROUGE-L F1: LCS length relative to both sequences."""
    if not a or not b:
        return float(a == b)
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v))
    nu = sum(x * x for x in u) ** 0.5
    nv = sum(y * y for y in v) ** 0.5
    return dot / (nu * nv) if nu and nv else 0.0


def sentence_transformer_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    from sentence_transformers import SentenceTransformer  # heavy import, only when used

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True).tolist()


class Scorer:
    def __init__(
        self,
        references: Sequence[str],
        embed: Optional[Embedder] = None,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.7,
    ):
        refs = [r for r in references if r.strip()]
        if not refs:
            raise ValueError("need at least one non-empty reference text")
        w = dict(weights or DEFAULT_WEIGHTS)
        if embed is None:  # lexical-only mode: drop semantic, renormalise
            w.pop("semantic", None)
        unknown = set(w) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown weight(s): {sorted(unknown)}")
        total = sum(w.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self.weights = {k: v / total for k, v in w.items()}
        self.threshold = threshold
        self.references = refs
        self._embed = embed
        self._ref_tokens = [tokenize(r) for r in refs]
        self._ref_vecs = embed(refs) if embed else None

    def score_many(self, texts: Sequence[str]) -> List[dict]:
        vecs = self._embed(texts) if self._embed else [None] * len(texts)
        return [self._score(t, v) for t, v in zip(texts, vecs)]

    def score(self, text: str) -> dict:
        return self.score_many([text])[0]

    def _score(self, text: str, vec) -> dict:
        toks = tokenize(text)
        metrics = {
            "rouge_l": max(rouge_l(toks, r) for r in self._ref_tokens),
            "jaccard": max(jaccard(toks, r) for r in self._ref_tokens),
        }
        best = None
        if vec is not None:
            sims = [cosine(vec, r) for r in self._ref_vecs]
            best = max(range(len(sims)), key=sims.__getitem__)
            metrics["semantic"] = max(0.0, sims[best])  # clamp: opposite meaning scores 0, not negative
        score = sum(self.weights[k] * metrics[k] for k in self.weights)
        out = {k: round(v, 4) for k, v in metrics.items()}
        out["score"] = round(score, 4)
        out["passed"] = score >= self.threshold
        if best is not None:
            out["closest_reference"] = best
        return out


def summarize(results: Iterable[dict]) -> dict:
    results = list(results)
    n = len(results)
    if not n:
        return {"count": 0, "pass_rate": 0.0, "mean_score": 0.0, "min_score": 0.0}
    scores = [r["score"] for r in results]
    return {
        "count": n,
        "pass_rate": round(sum(r["passed"] for r in results) / n, 4),
        "mean_score": round(sum(scores) / n, 4),
        "min_score": min(scores),
    }


# ---------- I/O ----------

def read_texts(path: str, field: str = "text") -> List[dict]:
    """.jsonl -> one object per line (needs `field`); anything else -> one text per non-blank line."""
    stream = sys.stdin if path == "-" else open(path, encoding="utf-8")
    with stream:
        lines = [ln.rstrip("\n") for ln in stream]
    if path.endswith(".jsonl"):
        rows = []
        for i, ln in enumerate(lines, 1):
            if not ln.strip():
                continue
            obj = json.loads(ln)
            if field not in obj:
                raise ValueError(f"{path}:{i}: missing field {field!r}")
            obj.setdefault("id", i)
            rows.append(obj)
        return rows
    return [{"id": i, field: ln} for i, ln in enumerate(lines, 1) if ln.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="brandsim",
        description="Score candidate text against on-brand reference texts.",
    )
    p.add_argument("-r", "--references", required=True, help="reference file (.txt one per line, or .jsonl)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("-t", "--text", help="single candidate string")
    src.add_argument("-i", "--input", help="candidates file (.txt or .jsonl), '-' for stdin")
    p.add_argument("--field", default="text", help="text field name in .jsonl files (default: text)")
    p.add_argument("--threshold", type=float, default=0.7, help="per-item pass threshold (default: 0.7)")
    p.add_argument("--min-pass-rate", type=float, help="exit 1 if batch pass rate is below this (CI gate)")
    p.add_argument("--weights", help='JSON, e.g. \'{"semantic":0.7,"rouge_l":0.3}\'')
    p.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    p.add_argument("--lexical-only", action="store_true", help="skip embeddings (no model download)")
    p.add_argument("-o", "--output", help="write per-item results as JSONL here")
    p.add_argument("--version", action="version", version=__version__)
    a = p.parse_args(argv)

    refs = [r[a.field] for r in read_texts(a.references, a.field)]
    items = [{"id": 1, a.field: a.text}] if a.text is not None else read_texts(a.input, a.field)
    embed = None if a.lexical_only else sentence_transformer_embedder(a.model)
    scorer = Scorer(refs, embed=embed, weights=json.loads(a.weights) if a.weights else None, threshold=a.threshold)

    results = scorer.score_many([it[a.field] for it in items])
    for it, res in zip(items, results):
        res["id"] = it["id"]

    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")

    _print_table(results, items, a.field)
    summary = summarize(results)
    print(json.dumps({**summary, "threshold": a.threshold, "weights": scorer.weights}, indent=2))

    if a.min_pass_rate is not None and summary["pass_rate"] < a.min_pass_rate:
        print(f"FAIL: pass rate {summary['pass_rate']:.2%} < required {a.min_pass_rate:.2%}", file=sys.stderr)
        return 1
    return 0


def _print_table(results: List[dict], items: List[dict], field: str) -> None:
    cols = [c for c in ("semantic", "rouge_l", "jaccard") if c in results[0]] if results else []
    w = max([len("id")] + [len(str(r["id"])) for r in results]) + 2
    print(f"{'id':<{w}}" + "".join(f"{c:>10}" for c in cols) + f"{'score':>8}  ok   text")
    for res, it in zip(results, items):
        snippet = it[field].replace("\n", " ")
        snippet = snippet if len(snippet) <= 48 else snippet[:45] + "..."
        print(
            f"{str(res['id']):<{w}}"
            + "".join(f"{res[c]:>10.3f}" for c in cols)
            + f"{res['score']:>8.3f}  {'PASS' if res['passed'] else 'fail'} {snippet}"
        )


if __name__ == "__main__":
    sys.exit(main())
