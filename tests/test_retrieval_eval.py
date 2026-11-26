"""Unit tests for src.evals.retrieval_eval, with fake chunks, search and embeddings."""

import math
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.evals import retrieval_eval as ev

CHUNKS = [
    {"doc_name": f"doc{d}.pdf", "page_number": i // 2 + 1, "chunk_index": i % 2, "content": f"text {d} {i}"}
    for d in range(5)
    for i in range(22)
]


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name, file in (
        ("QUESTIONS_PATH", "q.jsonl"),
        ("EMBEDDINGS_PATH", "emb.jsonl"),
        ("RUNS_LOG", "runs.jsonl"),
        ("RUNS_DIR", "runs"),
        ("PDF_DIR", "pdfs"),
    ):
        monkeypatch.setattr(ev, name, tmp_path / file)
    llm = MagicMock(side_effect=lambda *a, **k: ev.Question(question=f"spm {llm.call_count}?"))
    embed = MagicMock(return_value=[0.1, 0.2])
    # The fake search always ranks doc0 page 1 first.
    found = [SimpleNamespace(doc_name="doc0.pdf", page_number=p, score=1 / p) for p in (1, 1, 2)]
    monkeypatch.setattr(ev, "load_chunks", lambda: CHUNKS)
    monkeypatch.setattr(ev, "generate_structured", llm)
    monkeypatch.setattr(ev, "embed_query", embed)
    monkeypatch.setattr(ev, "search", MagicMock(return_value=found))
    monkeypatch.setattr(ev, "git_state", lambda: ("abc123", []))
    return SimpleNamespace(llm=llm, embed=embed)


def make_questions(n=4, split="dev"):
    rows = [
        {"id": f"q{i}", "type": "messy_user", "question": f"spm {i}?", "gold_pages": ["doc0.pdf#p1"],
         "source_chunk": 0, "split": split, "checked": True, "dropped": False}
        for i in range(n)
    ]
    ev.write_jsonl(ev.QUESTIONS_PATH, rows)
    return rows


# --- metrics ---

def test_rank_pages_counts_a_page_once_at_its_best_rank():
    chunks = [SimpleNamespace(doc_name="a", page_number=p, score=s) for p, s in ((1, 0.9), (1, 0.8), (2, 0.7))]
    assert ev.rank_pages(chunks) == [("a#p1", 0.9), ("a#p2", 0.7)]


def test_score_hit_at_rank_two():
    m = ev.score_ranking(["x", "g", "y"], ["g"])
    assert (m["recall@1"], m["recall@5"], m["mrr@20"]) == (0.0, 1.0, 0.5)
    assert m["ndcg@5"] == pytest.approx(1 / math.log2(3))


def test_score_no_hit():
    assert set(ev.score_ranking(["x", "y"], ["g"]).values()) == {0.0}


def test_score_two_gold_pages():
    m = ev.score_ranking(["g1", "x", "g2"], ["g1", "g2"])
    assert m["recall@1"] == 0.5
    assert m["recall@5"] == 1.0
    assert m["mrr@20"] == 1.0
    assert m["ndcg@5"] == pytest.approx((1 + 0.5) / (1 + 1 / math.log2(3)))


def test_time_stats():
    assert ev.time_stats(list(range(1, 101))) == {"median": 50.5, "p95": 95}


def test_compare_verdicts():
    def rows(values):
        return [{"id": str(i), "metrics": {"ndcg@5": v}} for i, v in enumerate(values)]

    clear = ev.compare(rows([1.0] * 20), rows([0.0] * 20))
    assert (clear["better"], clear["worse"], clear["verdict"]) == (20, 0, "better")
    mixed = ev.compare(rows([1.0, 0.0] * 10), rows([0.0, 1.0] * 10))
    assert mixed["verdict"] == "no measurable difference"


# --- generate ---

def test_generate_counts_and_no_repeats(env):
    ev.generate()
    rows = ev.read_jsonl(ev.QUESTIONS_PATH)
    assert Counter(r["type"] for r in rows) == {t: 30 for t in ev.STYLES}
    assert Counter((r["type"], r["split"]) for r in rows if r["split"] == "test") == {(t, "test"): 10 for t in ev.STYLES}
    assert len({(r["gold_pages"][0], r["source_chunk"]) for r in rows}) == 90
    assert set(Counter(r["gold_pages"][0].split("#")[0] for r in rows).values()) == {18}
    with pytest.raises(SystemExit):
        ev.generate()
    assert env.llm.call_count == 90


def test_generate_resumes_without_repeating_calls(env):
    answers = [ev.Question(question=f"spm {i}?") for i in range(90)]
    env.llm.side_effect = [*answers[:40], RuntimeError("quota")]
    with pytest.raises(RuntimeError):
        ev.generate()
    assert not ev.QUESTIONS_PATH.exists()
    env.llm.reset_mock()
    env.llm.side_effect = answers[40:]
    ev.generate()
    assert env.llm.call_count == 50
    assert len(ev.read_jsonl(ev.QUESTIONS_PATH)) == 90


# --- check ---

def test_check_saves_and_continues(env, monkeypatch):
    rows = make_questions(3, split="test")
    for r in rows:
        r["checked"] = False
    ev.write_jsonl(ev.QUESTIONS_PATH, rows)
    answers = iter(["a", "doc0.pdf#p2", "f", "k", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    ev.check()
    saved = ev.read_jsonl(ev.QUESTIONS_PATH)
    assert [r["checked"] for r in saved] == [True, False, False]
    assert saved[0]["gold_pages"] == ["doc0.pdf#p1", "doc0.pdf#p2"]
    answers = iter(["d", "k"])
    ev.check()
    saved = ev.read_jsonl(ev.QUESTIONS_PATH)
    assert [(r["checked"], r["dropped"]) for r in saved] == [(True, False), (True, True), (True, False)]


# --- run ---

def test_run_logs_and_reuses_embeddings(env):
    make_questions()
    first = ev.run("vector")
    second = ev.run("hybrid", baseline=first["run_id"])
    assert env.embed.call_count == 4
    assert [r["run_id"] for r in ev.read_jsonl(ev.RUNS_LOG)] == [first["run_id"], second["run_id"]]
    assert len(ev.read_jsonl(ev.RUNS_DIR / f"{first['run_id']}.jsonl")) == 4
    assert first["commit"] == "abc123"
    assert first["metrics"]["recall@1"] == 1.0
    assert first["chunk_count"] == len(CHUNKS)


def test_run_skips_dropped_and_warns_unchecked(env, capsys):
    rows = make_questions()
    rows[0]["dropped"] = True
    rows[1]["checked"] = False
    ev.write_jsonl(ev.QUESTIONS_PATH, rows)
    assert ev.run("vector")["questions"] == 3
    assert "1 of 3" in capsys.readouterr().out


def test_run_rejects_uncommitted_changes(env, monkeypatch):
    make_questions()
    monkeypatch.setattr(ev, "git_state", lambda: ("abc123", ["src/x.py"]))
    with pytest.raises(SystemExit):
        ev.run("vector")
    assert env.embed.call_count == 0
    assert ev.run("vector", allow_dirty=True)["dirty"] is True


def test_run_rejects_mismatched_baseline(env):
    rows = make_questions()
    base = ev.run("vector")["run_id"]
    rows[0]["gold_pages"] = ["doc0.pdf#p2"]
    ev.write_jsonl(ev.QUESTIONS_PATH, rows)
    with pytest.raises(SystemExit):
        ev.run("hybrid", baseline=base)
    assert len(ev.read_jsonl(ev.RUNS_LOG)) == 1
