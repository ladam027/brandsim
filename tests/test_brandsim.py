import json
import os
import subprocess
import sys
import zlib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import brandsim as bs  # noqa: E402


def bow_embed(texts, dim=64):
    """Deterministic stand-in for a sentence model: hashed bag of words."""
    out = []
    for t in texts:
        v = [0.0] * dim
        for tok in bs.tokenize(t):
            v[zlib.crc32(tok.encode()) % dim] += 1.0
        out.append(v)
    return out


REFS = ["Thanks for reaching out, happy to help with your order!", "Orders ship within two business days."]


def test_tokenize():
    assert bs.tokenize("We're HAPPY to help! 2 days.") == ["we're", "happy", "to", "help", "2", "days"]


def test_rouge_l_known_values():
    assert bs.rouge_l("a b c d".split(), "a b c d".split()) == 1.0
    assert bs.rouge_l("a b c".split(), "x y z".split()) == 0.0
    # LCS("a b c d", "a c d e") = 3 -> P=3/4, R=3/4
    assert bs.rouge_l("a b c d".split(), "a c d e".split()) == pytest.approx(0.75)
    assert bs.rouge_l([], []) == 1.0 and bs.rouge_l([], ["a"]) == 0.0


def test_jaccard_and_cosine():
    assert bs.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert bs.cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert bs.cosine([1, 0], [0, 1]) == 0.0
    assert bs.cosine([0, 0], [1, 1]) == 0.0


def test_exact_match_scores_one_and_picks_closest_reference():
    s = bs.Scorer(REFS, embed=bow_embed)
    r = s.score(REFS[1])
    assert r["score"] == pytest.approx(1.0) and r["passed"]
    assert r["closest_reference"] == 1


def test_ranking_on_brand_beats_off_topic():
    s = bs.Scorer(REFS, embed=bow_embed)
    near, far = s.score_many(["Thanks for reaching out, glad to help with the order!", "Quarterly earnings beat estimates."])
    assert near["score"] > 0.5 > far["score"]
    assert not far["passed"]


def test_lexical_only_drops_semantic_and_renormalises():
    s = bs.Scorer(REFS)
    assert "semantic" not in s.weights
    assert sum(s.weights.values()) == pytest.approx(1.0)
    assert "semantic" not in s.score("hello")


def test_threshold_controls_pass():
    text = "Thanks for reaching out about shipping."
    assert bs.Scorer(REFS, threshold=0.0).score(text)["passed"]
    assert not bs.Scorer(REFS, threshold=0.99).score(text)["passed"]


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        bs.Scorer(["", "  "])
    with pytest.raises(ValueError):
        bs.Scorer(REFS, weights={"vibes": 1.0})
    with pytest.raises(ValueError):
        bs.Scorer(REFS, weights={"rouge_l": 0, "jaccard": 0})


def test_summarize():
    res = [{"score": 0.9, "passed": True}, {"score": 0.5, "passed": False}]
    assert bs.summarize(res) == {"count": 2, "pass_rate": 0.5, "mean_score": 0.7, "min_score": 0.5}
    assert bs.summarize([])["count"] == 0


def test_cli_batch_and_ci_gate(tmp_path):
    out = tmp_path / "out.jsonl"
    base = [sys.executable, os.path.join(ROOT, "brandsim.py"), "--lexical-only",
            "-r", os.path.join(ROOT, "examples", "references.txt"),
            "-i", os.path.join(ROOT, "examples", "candidates.jsonl")]
    ok = subprocess.run(base + ["-o", str(out), "--threshold", "0.0"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["id"] for r in rows][:2] == ["on-brand-1", "on-brand-2"] and len(rows) == 5

    gated = subprocess.run(base + ["--threshold", "0.99", "--min-pass-rate", "0.95"], capture_output=True, text=True)
    assert gated.returncode == 1 and "FAIL" in gated.stderr


@pytest.mark.skipif(not os.environ.get("BRANDSIM_REAL_MODEL"), reason="set BRANDSIM_REAL_MODEL=1 to run with the real model")
def test_real_model_separates_paraphrase_from_off_topic():
    s = bs.Scorer(REFS, embed=bs.sentence_transformer_embedder())
    para, off = s.score_many(["Glad to assist with your purchase, thanks for contacting us!", "The weather is cold today."])
    assert para["semantic"] > off["semantic"] + 0.2
