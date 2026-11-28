# Pilot retrieval results

Run on 2026-11-27, commit `78fe373`, 60 `dev` questions, 209 chunks.

This run used scorer version 1, which marked results by page. It had two flaws: identical chunks got equal scores in arbitrary order, and some pages that answer a question weren't marked correct. Scorer version 2 replaces it, so don't compare these numbers with later runs. See [Improving retrieval](../../README.md#improving-retrieval).

## What was evaluated

| Method | How it searches | Evaluated |
| :--- | :--- | :---: |
| `vector` | By meaning: cosine similarity of `gemini-embedding-2-preview` embeddings (768 dimensions) | Yes |
| `hybrid` | `vector` plus PostgreSQL Norwegian full-text search (not BM25), merged with Reciprocal Rank Fusion | Yes |
| Keyword search alone | | No |
| BM25 | | No (step 1) |

Only retrieval was measured. No answers were generated and no judge was used.

## Logged results

| Method | Hit@1 | Hit@5 | Hit@20 | MRR@20 | nDCG@5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `vector` | 0.617 | 0.950 | 1.000 | 0.765 | 0.807 |
| `hybrid` | 0.683 | 0.950 | 1.000 | 0.805 | 0.838 |

## The logged gain of `hybrid` is overstated

- Keyword search matched only 2 of the 60 questions (`q043` and `q092`). It requires every word of the question to be in the chunk.
- 4 questions scored better with `hybrid`. Only those 2 come from keyword search.
- The other 2 (`q026`, `q071`) come from ties. Some chunks have identical embeddings, and the two searches order them differently.

Counting only the keyword matches:

| Method | Hit@1 | MRR@20 | nDCG@5 |
| :--- | ---: | ---: | ---: |
| `vector` | 0.617 | 0.765 | 0.807 |
| `hybrid` | 0.650 | 0.786 | 0.823 |

On nDCG@5: 2 better, 0 worse, mean difference +0.016, 95% interval 0.000 to +0.041. This is no measurable difference.

## What is saved

| Data | Saved | Where |
| :--- | :---: | :--- |
| Metrics, commit, data hashes, package versions per run | Yes | `results/retrieval_runs.jsonl` |
| Correct pages, ranked pages with scores, search time per question | Yes | `results/runs/` |
| Question texts and correct pages | Yes | `data/retrieval_eval.jsonl` |
| Question embeddings | Yes | `data/query_embeddings.jsonl` |
| Text of the retrieved chunks (the context) | No | |
| Chunk IDs | No | |
| Answers | No | None were generated |

To see the context again, the database must hold the same chunks. Each run stores a hash of the chunk texts to check this.

## Limits

- The questions were generated from the policy text, so real questions are probably harder.
- The `dev` questions are not hand-checked.
- The `test` questions are not scored.
