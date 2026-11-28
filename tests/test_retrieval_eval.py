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
    found = [
        SimpleNamespace(id=i, doc_name="doc0.pdf", page_number=1, score=1 / i, content=text)
        for i, text in ((1, "Her står  Svaret."), (2, "noe annet"))
    ]
    monkeypatch.setattr(ev, "load_chunks", lambda: CHUNKS)
    monkeypatch.setattr(ev, "generate_structured", llm)
    monkeypatch.setattr(ev, "embed_query", embed)
    monkeypatch.setattr(ev, "search", MagicMock(return_value=found))
    monkeypatch.setattr(ev, "git_state", lambda: ("abc123", []))
    return SimpleNamespace(llm=llm, embed=embed)


def make_questions(n=4, split="dev"):
    rows = [
        {"id": f"q{i}", "type": "messy_user", "question": f"spm {i}?", "gold_pages": ["doc0.pdf#p1"],
         "source_chunk": 0, "split": split, "checked": True, "dropped": False, "answer": "står svaret"}
        for i in range(n)
    ]
    ev.write_jsonl(ev.QUESTIONS_PATH, rows)
    return rows


# --- metrics ---

def test_contains_answer_ignores_case_and_spacing():
    assert ev.contains_answer("Egenandelen  er\n4 000 kroner.", "egenandelen er 4 000")
    assert not ev.contains_answer("Egenandelen er 2 000 kroner.", "egenandelen er 4 000")


def test_score_hit_at_rank_two():
    m = ev.score_ranking([False, True, False])
    assert (m["hit@1"], m["hit@5"], m["mrr@20"]) == (0.0, 1.0, 0.5)
    assert m["ndcg@5"] == pytest.approx(1 / math.log2(3))


def test_score_no_hit():
    assert set(ev.score_ranking([False, False]).values()) == {0.0}


def test_score_only_the_first_correct_result_counts():
    assert ev.score_ranking([False, True, True]) == ev.score_ranking([False, True, False])


def test_score_hit_below_rank_five():
    m = ev.score_ranking([False] * 5 + [True])
    assert (m["hit@5"], m["hit@20"], m["ndcg@5"]) == (0.0, 1.0, 0.0)


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
    answers = iter(["f", "k", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    ev.check()
    saved = ev.read_jsonl(ev.QUESTIONS_PATH)
    assert [r["checked"] for r in saved] == [True, False, False]
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
    assert first["metrics"]["hit@1"] == 1.0
    assert first["chunk_count"] == len(CHUNKS)
    assert first["scorer"] == 2
    row = ev.read_jsonl(ev.RUNS_DIR / f"{first['run_id']}.jsonl")[0]
    assert [c["correct"] for c in row["ranked"]] == [True, False]


def test_run_needs_an_answer_sentence(env):
    rows = make_questions()
    del rows[0]["answer"]
    ev.write_jsonl(ev.QUESTIONS_PATH, rows)
    with pytest.raises(SystemExit):
        ev.run("vector")
    assert env.embed.call_count == 0


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
    rows[0]["answer"] = "noe annet"
    ev.write_jsonl(ev.QUESTIONS_PATH, rows)
    with pytest.raises(SystemExit):
        ev.run("hybrid", baseline=base)
    assert len(ev.read_jsonl(ev.RUNS_LOG)) == 1


# --- wording ---

def test_copies_source_finds_four_words_in_a_row():
    source = "Tyveri av sykkel registrert i FG-godkjent sykkelregister 2 000"
    assert ev.copies_source("Hva får jeg ved tyveri av sykkel registrert der?", source)
    assert not ev.copies_source("Hva får jeg hvis sykkelen min blir stjålet?", source)
    assert not ev.copies_source("tyveri av sykkel", source)


def test_run_reworded_searches_with_the_reworded_text(env):
    rows = make_questions()
    rows[0]["reworded"] = "ny ordlyd?"
    ev.write_jsonl(ev.QUESTIONS_PATH, rows)
    original = ev.run("vector")
    reworded = ev.run("vector", wording="reworded")
    assert ev.search.call_args_list[4].args[0] == "ny ordlyd?"
    assert ev.search.call_args_list[5].args[0] == "spm 1?"
    assert (original["wording"], reworded["wording"]) == ("original", "reworded")
    assert reworded["run_id"].endswith("-reworded")
    assert original["questions_hash"] != reworded["questions_hash"]
    with pytest.raises(SystemExit):
        ev.run("vector", wording="reworded", baseline=original["run_id"])


def test_run_rejects_a_reworded_question_that_copies_the_source(env, monkeypatch):
    chunks = [dict(CHUNKS[0], content="egenandelen er 4 000 kroner"), *CHUNKS[1:]]
    monkeypatch.setattr(ev, "load_chunks", lambda: chunks)
    rows = make_questions()
    rows[0]["reworded"] = "er egenandelen er 4 000 nå?"
    ev.write_jsonl(ev.QUESTIONS_PATH, rows)
    with pytest.raises(SystemExit):
        ev.run("vector", wording="reworded")
    assert env.embed.call_count == 0
