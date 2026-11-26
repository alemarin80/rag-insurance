"""Retrieval-only evaluation: no answer model and no judge.

Spec: docs/specs/step0-retrieval-eval.md

Usage:
    python -m src.evals.retrieval_eval generate
    python -m src.evals.retrieval_eval check [--split test|dev|all]
    python -m src.evals.retrieval_eval run --mode vector|hybrid [--split dev|test]
        [--baseline RUN_ID] [--allow-dirty]
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel

from src.ingestion.db import get_conn
from src.ingestion.ingest import CHUNK_OVERLAP, CHUNK_SIZE
from src.llm import EMBED_DIM, EMBED_MODEL, generate_structured, resolve
from src.retrieval.search import embed_query, search

QUESTIONS_PATH = Path("data/retrieval_eval.jsonl")
EMBEDDINGS_PATH = Path("data/query_embeddings.jsonl")
RUNS_LOG = Path("results/retrieval_runs.jsonl")
RUNS_DIR = Path("results/runs")
PDF_DIR = Path("data/policies")

SEED = 42
PER_TYPE = 30
TEST_PER_TYPE = 10
TOP_K = 20
BOOTSTRAP_SAMPLES = 10_000
METRICS = ("recall@1", "recall@5", "recall@20", "mrr@20", "ndcg@5")
PACKAGES = ("google-genai", "pgvector", "psycopg", "pymupdf")

# Same instructions as generate_eval_set() in run_eval.py.
STYLES = {
    "exact_lookup": "Ask a specific question targeting an exact number, age, day limit, kroner amount, or clause condition in the excerpt—without copying the surrounding sentence.",
    "messy_user": "Write an informal, slightly vague real-life scenario from a stressed Norwegian customer (no insurance jargon) that this excerpt resolves.",
    "exclusion_edge": "Ask an edge-case question testing whether a specific exception or limitation in the excerpt applies.",
}


class Question(BaseModel):
    """One simulated customer question."""

    question: str


# --- files and database ---

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Replace the file in one step, so an interrupted write can't truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_chunks() -> list[dict]:
    """Return every chunk in a stable order."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT doc_name, page_number, chunk_index, content FROM chunks "
            "ORDER BY doc_name, page_number, chunk_index;"
        ).fetchall()
    return [dict(zip(("doc_name", "page_number", "chunk_index", "content"), r)) for r in rows]


def page_id(chunk: dict) -> str:
    return f"{chunk['doc_name']}#p{chunk['page_number']}"


# --- generate ---

def sample_chunks(chunks: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """Pick n different chunks, taking one document at a time so documents stay even."""
    if len(chunks) < n:
        raise SystemExit(f"Need {n} chunks, found {len(chunks)}.")
    rng = random.Random(seed)
    docs: dict[str, list[dict]] = {}
    for c in chunks:
        docs.setdefault(c["doc_name"], []).append(c)
    for pool in docs.values():
        rng.shuffle(pool)
    picked: list[dict] = []
    while len(picked) < n:
        for name in sorted(docs):
            if docs[name] and len(picked) < n:
                picked.append(docs[name].pop())
    return picked


def plan_questions(chunks: list[dict]) -> list[tuple[dict, str]]:
    """Return (row without question, source text) pairs. Deterministic for the same chunks."""
    types = list(STYLES)
    picked = sample_chunks(chunks, PER_TYPE * len(types))
    plan = []
    for i, c in enumerate(picked):
        row = {
            "id": f"q{i + 1:03d}",
            "type": types[i % len(types)],
            "question": None,
            "gold_pages": [page_id(c)],
            "source_chunk": c["chunk_index"],
            "split": "dev",
            "checked": False,
            "dropped": False,
        }
        plan.append((row, c["content"]))
    rng = random.Random(SEED)
    for t in types:
        for row in rng.sample([r for r, _ in plan if r["type"] == t], TEST_PER_TYPE):
            row["split"] = "test"
    return plan


def generate() -> None:
    if QUESTIONS_PATH.exists():
        raise SystemExit(f"{QUESTIONS_PATH} exists. It is generated only once.")
    partial = QUESTIONS_PATH.with_name(QUESTIONS_PATH.stem + ".partial.jsonl")
    done = {r["id"]: r for r in read_jsonl(partial)}
    plan = plan_questions(load_chunks())
    provider, model = resolve("sim")
    print(f"{len(plan) - len(done)} LLM calls with {provider}/{model} ({len(done)} already saved).")
    for row, text in plan:
        saved = done.get(row["id"])
        if saved:
            if (saved["gold_pages"], saved["source_chunk"]) != (row["gold_pages"], row["source_chunk"]):
                raise SystemExit(f"The chunks changed since {partial} was written. Delete it to start again.")
            continue
        prompt = (
            f"You are a Norwegian insurance customer. {STYLES[row['type']]}\n"
            f"Write ONE question in Norwegian.\n\nExcerpt:\n{text}"
        )
        row["question"] = generate_structured(prompt, Question, role="sim", temperature=0.7).question
        append_jsonl(partial, row)
        done[row["id"]] = row
        print(f"{row['id']} {row['type']}")
    write_jsonl(QUESTIONS_PATH, [done[row["id"]] for row, _ in plan])
    partial.unlink(missing_ok=True)
    print(f"Wrote {len(plan)} questions to {QUESTIONS_PATH}.")


# --- check ---

def show(text: str) -> None:
    print(textwrap.fill(text, width=90))


def check(split: str = "test") -> None:
    """Hand-check questions one at a time and save after each answer."""
    questions = read_jsonl(QUESTIONS_PATH)
    if not questions:
        raise SystemExit(f"No questions. Run generate first ({QUESTIONS_PATH}).")
    chunks = load_chunks()
    pages: dict[str, list[dict]] = {}
    for c in chunks:
        pages.setdefault(page_id(c), []).append(c)
    todo = [q for q in questions if not q["checked"] and split in (q["split"], "all")]
    for n, q in enumerate(todo, start=1):
        print(f"\n=== {n}/{len(todo)}  {q['id']}  {q['type']} ===")
        show(q["question"])
        source = q["gold_pages"][0]
        print(f"\n--- source chunk ({source}) ---")
        match = [c for c in pages.get(source, []) if c["chunk_index"] == q["source_chunk"]]
        show(match[0]["content"] if match else "Source chunk not found. Press f to see the page.")
        while True:
            answer = input("\n[k]eep  [d]rop  [a]dd page  [f]ull pages  [q]uit: ").strip().lower()
            if answer == "q":
                return
            if answer == "f":
                for p in q["gold_pages"]:
                    print(f"\n--- {p} ---")
                    for c in pages.get(p, []):
                        show(c["content"])
                        print()
            elif answer == "a":
                new = input("Page (document.pdf#pN): ").strip()
                if new not in pages:
                    print("No such page.")
                elif new not in q["gold_pages"]:
                    q["gold_pages"].append(new)
                    write_jsonl(QUESTIONS_PATH, questions)
                    print("Added. Press k to keep the question.")
            elif answer in ("k", "d"):
                q["checked"], q["dropped"] = True, answer == "d"
                write_jsonl(QUESTIONS_PATH, questions)
                break
    print(f"\nNo unchecked questions left in split '{split}'.")


# --- metrics ---

def rank_pages(chunks: list) -> list[tuple[str, float]]:
    """Return unique pages with their score, at the position of their best chunk."""
    ranked: dict[str, float] = {}
    for c in chunks:
        ranked.setdefault(f"{c.doc_name}#p{c.page_number}", float(c.score))
    return list(ranked.items())


def score_ranking(ranked: list[str], gold: list[str]) -> dict[str, float]:
    """Score a page ranking without repeats against the correct pages (binary relevance)."""
    relevant = set(gold)
    out = {f"recall@{k}": len(relevant & set(ranked[:k])) / len(relevant) for k in (1, 5, 20)}
    first = next((i for i, p in enumerate(ranked[:20], start=1) if p in relevant), None)
    out["mrr@20"] = 1 / first if first else 0.0
    dcg = sum(1 / math.log2(i + 1) for i, p in enumerate(ranked[:5], start=1) if p in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), 5) + 1))
    out["ndcg@5"] = dcg / ideal
    return out


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {m: round(statistics.fmean(r["metrics"][m] for r in rows), 4) for m in METRICS}


def time_stats(values: list[float]) -> dict[str, float]:
    """Median and 95th percentile (nearest rank) in milliseconds."""
    ordered = sorted(values)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {"median": round(statistics.median(ordered), 1), "p95": round(p95, 1)}


def compare(new: list[dict], old: list[dict]) -> dict:
    """Paired comparison on nDCG@5 with a 95% bootstrap interval of the mean difference."""
    before = {r["id"]: r["metrics"]["ndcg@5"] for r in old}
    diffs = [r["metrics"]["ndcg@5"] - before[r["id"]] for r in new]
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(BOOTSTRAP_SAMPLES))
    low = means[int(0.025 * BOOTSTRAP_SAMPLES)]
    high = means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]
    if low <= 0 <= high:
        verdict = "no measurable difference"
    else:
        verdict = "better" if low > 0 else "worse"
    return {
        "better": sum(d > 1e-9 for d in diffs),
        "worse": sum(d < -1e-9 for d in diffs),
        "mean_difference": statistics.fmean(diffs),
        "interval": (low, high),
        "verdict": verdict,
    }


# --- run ---

def git_state() -> tuple[str, list[str]]:
    """Return the commit and the changed files that make a run irreproducible."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    commit = git("rev-parse", "HEAD").strip()
    changed = [line[3:].split(" -> ")[-1] for line in git("status", "--porcelain", "--untracked-files=all").splitlines()]
    ignored = ("results/", EMBEDDINGS_PATH.as_posix())
    return commit, [p for p in changed if not p.startswith(ignored)]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_embeddings(questions: list[dict]) -> dict[str, tuple[list[float], float]]:
    """Return {question: (vector, ms)}, calling the API only for questions not saved yet."""
    saved = {
        r["question"]: (r["vector"], r["ms"])
        for r in read_jsonl(EMBEDDINGS_PATH)
        if (r["model"], r["dim"]) == (EMBED_MODEL, EMBED_DIM)
    }
    for q in questions:
        text = q["question"]
        if text in saved:
            continue
        t0 = time.perf_counter()
        vector = [float(x) for x in embed_query(text)]
        ms = round((time.perf_counter() - t0) * 1000, 1)
        append_jsonl(EMBEDDINGS_PATH, {"model": EMBED_MODEL, "dim": EMBED_DIM, "question": text, "ms": ms, "vector": vector})
        saved[text] = (vector, ms)
    return saved


def run(mode: str, split: str = "dev", baseline: str | None = None, allow_dirty: bool = False) -> dict:
    commit, changed = git_state()
    if changed and not allow_dirty:
        raise SystemExit("Uncommitted changes (commit them or pass --allow-dirty):\n  " + "\n  ".join(changed))
    questions = [q for q in read_jsonl(QUESTIONS_PATH) if q["split"] == split and not q["dropped"]]
    if not questions:
        raise SystemExit(f"No questions in split '{split}' of {QUESTIONS_PATH}.")
    unchecked = sum(not q["checked"] for q in questions)
    if unchecked:
        print(f"Warning: {unchecked} of {len(questions)} questions in '{split}' are not hand-checked.")
    used = [[q["id"], q["question"], sorted(q["gold_pages"])] for q in questions]
    questions_hash = sha256(json.dumps(used, ensure_ascii=False).encode())

    base = None
    if baseline:  # check before any API call
        base = next((r for r in read_jsonl(RUNS_LOG) if r["run_id"] == baseline), None)
        if base is None:
            raise SystemExit(f"No run '{baseline}' in {RUNS_LOG}.")
        if (base["questions_hash"], base["split"]) != (questions_hash, split):
            raise SystemExit("The baseline used different questions or another split.")

    chunks = load_chunks()
    embeddings = get_embeddings(questions)
    results = []
    for q in questions:
        vector, _ = embeddings[q["question"]]
        t0 = time.perf_counter()
        found = search(q["question"], mode=mode, top_k=TOP_K, query_vector=vector)
        db_ms = (time.perf_counter() - t0) * 1000
        ranked = rank_pages(found)
        results.append({
            "id": q["id"],
            "type": q["type"],
            "gold_pages": q["gold_pages"],
            "ranked": [{"page": p, "score": round(s, 4)} for p, s in ranked],
            "metrics": score_ranking([p for p, _ in ranked], q["gold_pages"]),
            "db_ms": round(db_ms, 1),
        })

    now = datetime.now(timezone.utc)
    run_id = f"{now:%Y%m%dT%H%M%S%fZ}-{mode}-{split}"
    by_type = {t: [r for r in results if r["type"] == t] for t in STYLES}
    record = {
        "run_id": run_id,
        "time": now.isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": bool(changed),
        "method": mode,
        "chunking": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "embedding": {"model": EMBED_MODEL, "dim": EMBED_DIM},
        "questions_hash": questions_hash,
        "split": split,
        "questions": len(results),
        "unchecked": unchecked,
        "metrics": mean_metrics(results),
        "metrics_by_type": {t: mean_metrics(rows) for t, rows in by_type.items() if rows},
        "db_ms": time_stats([r["db_ms"] for r in results]),
        "embed_ms": time_stats([embeddings[q["question"]][1] for q in questions]),
        "pdf_sha256": {p.name: sha256(p.read_bytes()) for p in sorted(PDF_DIR.glob("*.pdf"))},
        "chunks_sha256": sha256(json.dumps(chunks, ensure_ascii=False).encode()),
        "chunk_count": len(chunks),
        "packages": {name: version(name) for name in PACKAGES},
    }
    write_jsonl(RUNS_DIR / f"{run_id}.jsonl", results)
    append_jsonl(RUNS_LOG, record)
    report(record)
    if base:
        c = compare(results, read_jsonl(RUNS_DIR / f"{baseline}.jsonl"))
        print(
            f"\nAgainst {baseline} on nDCG@5: {c['better']} better, {c['worse']} worse, "
            f"mean difference {c['mean_difference']:+.3f} "
            f"(95% interval {c['interval'][0]:+.3f} to {c['interval'][1]:+.3f}): {c['verdict']}"
        )
    return record


def report(record: dict) -> None:
    print(f"\nRun {record['run_id']}  ({record['questions']} questions)")
    print(f"{'type':16}" + "".join(f"{m:>11}" for m in METRICS))
    for name, values in [*record["metrics_by_type"].items(), ("all", record["metrics"])]:
        print(f"{name:16}" + "".join(f"{values[m]:11.3f}" for m in METRICS))
    for label, key in (("database search", "db_ms"), ("query embedding", "embed_ms")):
        print(f"{label}: median {record[key]['median']} ms, p95 {record[key]['p95']} ms")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.evals.retrieval_eval", description="Retrieval-only evaluation.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="write the question file once (paid LLM calls)")
    p = sub.add_parser("check", help="hand-check questions")
    p.add_argument("--split", choices=("test", "dev", "all"), default="test")
    p = sub.add_parser("run", help="score one retrieval method")
    p.add_argument("--mode", choices=("vector", "hybrid"), required=True)
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--baseline", metavar="RUN_ID", help="compare with an earlier run")
    p.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes and mark the run")
    args = parser.parse_args(argv)

    if args.command == "generate":
        generate()
    elif args.command == "check":
        check(args.split)
    else:
        run(args.mode, args.split, args.baseline, args.allow_dirty)


if __name__ == "__main__":
    sys.exit(main())
