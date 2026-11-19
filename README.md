# brandsim

Score how closely a piece of text matches a set of on-brand reference texts.

Built as an eval-harness piece for LLMs fine-tuned on brand voice: you give it
reference answers written the way the brand writes, you feed it model
generations, and it returns a per-item similarity score, a pass/fail against a
threshold, and a batch pass rate you can use to fail a CI run.

**Try it in the browser:** [ladam027.github.io/brandsim](https://ladam027.github.io/brandsim/).
Upload a small text sample and the page generates new sentences in the same voice,
then scores each one with the same metrics. It runs entirely client-side.

## How it scores

Each candidate is compared against every reference, and each metric keeps its
**best match**, so one strong reference is enough.

| Metric | What it captures | How |
|---|---|---|
| `semantic` | meaning and tone | cosine similarity of sentence embeddings (`all-MiniLM-L6-v2` by default) |
| `rouge_l` | phrasing and structure | ROUGE-L F1 over tokens (longest common subsequence) |
| `jaccard` | vocabulary and brand terms | token-set overlap |

`score = 0.6·semantic + 0.25·rouge_l + 0.15·jaccard` by default. The weights
are configurable, and they're renormalised so they always sum to 1.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

The embedding model (~90 MB) downloads on first use. Use `--lexical-only` to skip it.

## Usage

Score one string:

```bash
brandsim -r examples/references.txt -t "Happy to help! Orders ship in 2 business days."
```

Score a batch, write per-item results, and fail if fewer than 95% pass:

```bash
brandsim -r examples/references.txt -i examples/candidates.jsonl -o results.jsonl --min-pass-rate 0.95
```

Inputs are either `.txt` (one text per line) or `.jsonl` (one object per line
with a `text` field, plus an optional `id`; use `--field` to pick another field).
`-i -` reads from stdin.

| Flag | Default | Meaning |
|---|---|---|
| `--threshold` | `0.7` | per-item pass mark |
| `--min-pass-rate` | off | exit code 1 if the batch pass rate is below this |
| `--weights` | `{"semantic":0.6,"rouge_l":0.25,"jaccard":0.15}` | JSON metric weights |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | any sentence-transformers model |
| `--lexical-only` | off | no embeddings, lexical metrics only |
| `-o` | none | write per-item results as JSONL |

## Example output

From `examples/` (a fictional skincare brand's support answers):

```
id            semantic   rouge_l   jaccard   score  ok   text
on-brand-1       0.941     0.711     0.571   0.828  PASS Thanks for reaching out! The Everyday Serum i...
on-brand-2       0.925     0.542     0.469   0.760  PASS Happy to help! Your order ships within 2 busi...
paraphrase       0.754     0.186     0.175   0.525  fail So sorry the package showed up broken. A repl...
off-tone         0.237     0.143     0.079   0.190  fail Per policy, damaged goods claims require phot...
off-topic        0.041     0.067     0.037   0.047  fail Our quarterly earnings exceeded analyst expec...
{
  "count": 5,
  "pass_rate": 0.4,
  "mean_score": 0.47,
  "min_score": 0.0466,
  "threshold": 0.7,
  "weights": {"semantic": 0.6, "rouge_l": 0.25, "jaccard": 0.15}
}
FAIL: pass rate 40.00% < required 95.00%
```

The `paraphrase` row shows the trade-off the weights control. It says the
same thing as a reference in different words, so the semantic score is high
but the lexical scores are low. If you care about meaning and tone more than
wording, weight `semantic` up. With `--weights '{"semantic":1}'` the
paraphrase passes at 0.754, while the cold, policy-voiced `off-tone` answer
still fails at 0.237.

## Python API

```python
from brandsim import Scorer, sentence_transformer_embedder, summarize

scorer = Scorer(references, embed=sentence_transformer_embedder(), threshold=0.75)
results = scorer.score_many(model_outputs)
print(summarize(results))   # {'count': ..., 'pass_rate': ..., 'mean_score': ..., 'min_score': ...}
```

`embed` is any function that maps a list of strings to a list of vectors, so
you can plug in a different embedding model or API.

## Tests

```bash
pytest                              # fast: uses a deterministic stand-in embedder
BRANDSIM_REAL_MODEL=1 pytest        # also runs a check against the real model
```

## Limits

- Similarity to references is a proxy for brand alignment, not a guarantee. It
  rewards sounding like the references; it doesn't check facts. Pair it with a
  product-answer accuracy check.
- Scores depend on reference coverage. If no reference covers a topic, an
  on-brand answer about that topic will score low.
- Thresholds should be calibrated on a small human-labelled set before you
  rely on them.

## License

MIT
