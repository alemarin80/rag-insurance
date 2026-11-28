# Step 4: hybrid with BM25

Status: approved 2026-11-28.

## Goal

Combine `vector` and BM25 with Reciprocal Rank Fusion (RRF), and compare the result with today's `hybrid` (`vector` plus PostgreSQL keyword search) on the 53 `dev` questions.

On the reworded wording, `vector` and `bm25` each miss 11 questions in the top 5, but only 4 of the same ones. Fusing them could find more answers than either alone.

## Non-goals

- Changing today's `hybrid`. It stays as the baseline.
- Changing the API default. That's decided after the results.
- Tuning the RRF constant, the number of candidates or the BM25 settings.

## Design

- New mode `hybrid_bm25` in `search()`.
- The vector side is the same query as in `hybrid`: the top 30 chunks by cosine distance.
- The BM25 side is `bm25_search(query, top_k=30)`: up to 30 chunks with a score above 0.
- RRF as in `hybrid`: each chunk scores the sum of `1 / (60 + rank)` over the lists it appears in. Ties are broken by chunk id.
- The fusion runs in Python, in a function that takes two ranked lists of chunk ids, so it can be tested without a database.

## Evaluation

- `run --mode hybrid_bm25` on both wordings. It uses the cached query embeddings, so no API call.
- Main comparison, reworded wording: `--baseline 20261128T102405471181Z-hybrid-dev-reworded` (today's `hybrid`).
- Also compared on the reworded wording with `vector` (`20261128T102353496726Z-vector-dev-reworded`) and `bm25` (`20261128T103752492732Z-bm25-dev-reworded`), and on the original wording with today's `hybrid` (`20261128T084007943125Z-hybrid-dev`).
- The README lists the questions that got better and worse.

## Known flaws

- With 53 questions, only large differences are measurable.
- The RRF constant 60 and 30 candidates per side are untested defaults.

## Cost

None. Query embeddings are cached.

## Acceptance criteria

1. The fusion function gives hand-computed scores, ranks a chunk found by both sides above one found by one side at the same rank, and breaks ties by id.
1. `hybrid_bm25` with a precomputed query vector makes no embedding call.
1. A query with no BM25 match returns the vector ranking.
1. `vector`, `hybrid` and `bm25` give the same rankings as the logged runs.
1. The unit tests call no API and need no database.
