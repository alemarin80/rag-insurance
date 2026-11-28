# Step 0: retrieval-only evaluation

Status: approved 2026-11-27.

## Goal

Measure retrieval alone, well enough to tell whether a change to search or chunking helps. No answer model and no judge.

## Non-goals

- BM25, new chunking and reranking. They are later steps.
- Changes to `run_eval.py` or `data/eval_set.json`.
- Changes to search results. `search()` only gains an optional precomputed query vector.

## Question set

File: `data/retrieval_eval.jsonl`, committed. One question per line:

```json
{"id": "q001", "type": "messy_user", "question": "...", "gold_pages": ["fremtind_bil_kasko.pdf#p4"], "source_chunk": 1, "split": "dev", "checked": false, "dropped": false}
```

- 90 questions: 30 of each type (`exact_lookup`, `messy_user`, `exclusion_edge`), using the instructions in `run_eval.py`.
- Source chunks are spread evenly over the 5 PDFs, picked with a fixed seed, without repeats.
- No trap questions: they have no correct page.
- `gold_pages` starts with the source page. The hand-check can add pages.
- Split: 60 `dev`, 30 `test`, balanced by type, fixed seed.

## Hand-check

You review the 30 `test` questions. For each one the command shows the question and the source chunk, and the full gold pages on request. You choose: keep, drop, or add a gold page.

A dropped question is not replaced. The `dev` questions stay unchecked, so treat `dev` numbers as less reliable.

## Metrics

Search returns the top 20 chunks. Chunks from the same page count once, at their best rank. Metrics use this page ranking:

| Metric | Definition |
| :--- | :--- |
| Recall@1, @5, @20 | Share of the gold pages found in the top k |
| MRR@20 | 1 / rank of the first gold page, 0 if none |
| nDCG@5 | Binary relevance, divided by the best possible score |

Because pages count once, Recall@20 covers at most 20 pages and can cover fewer. These numbers aren't comparable with the Recall@5 in the README, which counts chunks.

Each metric is the mean over questions, in total and per type. Also report time per question (median and 95th percentile): database search in every run, and query embedding as measured when it was first computed.

## Comparing runs

`--baseline RUN_ID` compares two runs on the same questions:

- Number of questions that got better and worse on nDCG@5.
- Mean difference with a 95% bootstrap interval. If the interval includes 0, the output says "no measurable difference".

A comparison fails if the two runs used different questions or splits.

## Logs

- `results/retrieval_runs.jsonl`, committed: one line per run with run id, time, git commit, uncommitted-changes flag, method, chunking settings, embedding model and size, hash of the questions used, split, number of questions, metrics and times.
- `results/runs/RUN_ID.jsonl`, committed: per question, the gold pages, the ranked pages and the scores.

Files instead of a database table: they survive a database reset and show the history of improvements in Git.

## Reproducing a run

- `run` refuses to start if code has uncommitted changes, so the logged commit is the code that ran. Files under `results/` and the embedding cache don't count. `--allow-dirty` overrides this and marks the run.
- The PDFs and chunks aren't in Git. Each run line also stores a SHA-256 per PDF, a hash of the chunk texts, the chunk count and the versions of the retrieval packages.
- Query embeddings are saved in `data/query_embeddings.jsonl`, committed, keyed by model, size and question text. A question is embedded once. Later runs reuse the vector and make no API call.

## Commands

```bash
python -m src.evals.retrieval_eval generate            # write the question file
python -m src.evals.retrieval_eval check               # hand-check
python -m src.evals.retrieval_eval run --mode vector   # score on dev
python -m src.evals.retrieval_eval run --mode hybrid --split test --baseline RUN_ID
```

## Cost

- `generate`: one simulator LLM call per question (about 90), once. The simulator is set by `SIM_PROVIDER`. The question file is never regenerated in later steps.
- `run`: one query embedding per new question. Zero calls once cached.

## Acceptance criteria

1. `generate` writes 90 questions with 30 per type, a 60/30 split and no repeated source chunk. It refuses to overwrite an existing file.
1. `check` uses `test` by default, saves after each answer and continues where it stopped.
1. `generate` saves each question as it arrives. After a failure, the next call continues and repeats no LLM call.
1. Metric functions give the expected values for hand-computed cases, including a repeated page, no hit and two gold pages.
1. `run` uses `dev` by default, skips dropped questions and warns if unchecked questions remain in the split.
1. `run` appends one line to the run log and writes the per-question file.
1. A comparison reports better, worse, mean difference and interval, and rejects mismatched runs.
1. `run` rejects uncommitted code changes without `--allow-dirty`, and the run line holds the hashes and versions.
1. A second `run` on the same questions makes no embedding call.
1. The unit tests call no LLM API and need no database.

## First results

Run `vector` and `hybrid` on `dev` as the baseline for steps 1 to 3. Keep `test` for the final comparison.

## Proposed change: repeated text

Status: approved 2026-11-27. It replaces the parts above that conflict with it.

An AI assistant (Claude), not a person, checked most `test` questions: it judged whether each one can be answered and added pages that hold the same answer. The field `checked_by` records `user` or `assistant`. Treat the `test` labels as AI-checked.

As an exception to "not replaced", `q040`, `q019` and `q034` were dropped and replaced by `q091`, `q092` and `q093`, with one LLM call each. `q019` and `q034` came from chunks shorter than 12 words, so their replacements use chunks of at least 50 words.

The policies share text, so several pages can answer one question.

- Every page in `gold_pages` answers the question. Finding any one of them is a hit.
- A new command, `match`, adds pages to `gold_pages`. A page is added if it contains at least 80% of the source chunk's runs of 12 words, and every number in the source chunk. A chunk shorter than 12 words counts as one run. Running `match` again adds nothing new.
- `check` lists the added pages. `f` shows them in full, `a` adds a page and the new key `r` removes one.
- Metrics, with the rank of the first gold page found:

| Metric | Definition |
| :--- | :--- |
| Hit@1, @5, @20 (replaces Recall) | 1 if a gold page is in the top k, else 0 |
| MRR@20 | 1 / rank, 0 if none |
| nDCG@5 | 1 / log2(rank + 1), 0 if the rank is over 5 |

Limit: a page with less than 80% of the text is not added, even if it holds the answer. The hand-check can add such pages to `test` questions. `dev` questions keep the miss.

## Scorer version 2

Status: approved 2026-11-28. It replaces the parts above that conflict with it, including "Proposed change: repeated text".

Version 1 marked results by page. Identical chunks in different documents tied in arbitrary order, and pages that answered a question could be marked wrong.

- Each question has an `answer` field: one sentence or phrase copied from its source chunk, at most 200 characters (the chunk overlap). `answer_by` records who picked it. An AI assistant (Claude) picked all of them.
- A retrieved chunk is correct if it contains the answer, ignoring case and spacing, whichever document it comes from.
- Metrics use the chunk ranking, not a page ranking: Hit@1, @5, @20, MRR@20 and nDCG@5 of the first correct chunk.
- `gold_pages` holds only the source page. The `match` command is removed, and `check` shows the answer sentence.
- `run` refuses questions without an answer. The question hash covers the answers, so a comparison across different answers fails.
- Nine questions are dropped (`drop_reason`): five repeat another question's source text, and four can't be answered from their source text. This leaves 53 `dev` and 25 `test` questions.
- Each run line stores `scorer`. Runs without it are version 1.

