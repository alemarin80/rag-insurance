# Step 1: BM25 alone

Status: approved 2026-11-28.

## Goal

Add BM25 as a search method and compare it with `vector` and `hybrid` on the 53 `dev` questions, using scorer version 2.

## Non-goals

- Changing `hybrid` or the app's default search. Step 4 does that.
- Tuning BM25 settings.
- Splitting Norwegian compound words, such as `bilforsikring`.

## Design

- New module `src/retrieval/bm25.py`, using `bm25s` 0.3.16.
- Chunks and questions are processed alike: lowercase, remove Norwegian stop words, stem with the Norwegian Snowball stemmer (`PyStemmer` 3.1.0).
- Settings are the library defaults: method `lucene`, `k1` 1.5, `b` 0.75.
- The index is built in memory from the `chunks` table on first use, once per process. 209 chunks need no stored index.
- `search(query, mode="bm25")` returns only chunks that score above 0, so it can return fewer than `top_k`.
- An unknown `mode` raises an error. Today it runs `hybrid`.

## Evaluation

- `python -m src.evals.retrieval_eval run --mode bm25` and `run --mode bm25 --wording reworded` score on `dev`. They need no embedding and make no API call.
- Scoring is unchanged (version 2): a chunk is correct if it contains the question's answer sentence. If BM25 returns fewer than 20 chunks, the missing ranks count as wrong.
- The run line stores the BM25 settings and the versions of `bm25s` and `PyStemmer`. Its embedding fields are empty.
- Main baseline: the reworded `vector` run `20261128T102353496726Z-vector-dev-reworded`. Also compare on the original wording with `20261128T084006677909Z-vector-dev`.
- The README lists the questions that got better and worse, not only the means.

## Known flaws

- The original wording often reuses the policy's words, which favors BM25. The reworded wording shows BM25 without that help.
- With 53 questions, only large differences are measurable.

## Cost

None. No paid or API calls.

## Acceptance criteria

1. A question with another word form, such as `egenandelen` for `egenandel`, finds the chunk.
1. A question with no matching word returns an empty list.
1. `run --mode bm25` calls no embedding function.
1. Rerunning `vector` and `hybrid` on `dev` gives the same rankings as the logged runs.
1. The unit tests call no API and need no database.
