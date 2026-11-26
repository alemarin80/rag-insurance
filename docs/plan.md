# Retrieval improvement plan

Evaluation comes first. Each step changes one thing and logs one result.

| Step | Change | Needs new embeddings |
| :---: | :--- | :--- |
| 0 | Retrieval-only evaluation ([spec](specs/step0-retrieval-eval.md)) | Queries, once |
| 1 | BM25 alone, using `bm25s` with a Norwegian stemmer | No |
| 2 | Dense search alone | No |
| 3 | Hybrid with Reciprocal Rank Fusion (BM25 and dense) | No |
| 4 | Sentence chunking (spaCy Norwegian) against 1,200-character windows | All chunks |
| 5 | Reranker on the top 20 | No |
| 6 | Generation evaluation, with about 15 hand-graded answers | No |

## Rules

- The retrieval questions are generated once and committed. Later steps reuse them.
- Correct answers are stored as `document#page`, so they stay valid after re-chunking. Chunks must not cross a page boundary.
- Tune on the `dev` questions. Use the `test` questions for final numbers.
- Each logged run stores its Git commit.
- Don't run the ingest or `reset chunks` while the embedding API is unavailable. The ingest deletes all chunks before it calls the API.

## Decisions

- Embeddings stay on Gemini, using the free tier. The free tier can use the data you send to improve Google products.
- The model stays `gemini-embedding-2-preview`. Query embedding works with the current API key. The pricing page lists only `gemini-embedding-2`, and changing the model means re-embedding all chunks.
- The judge for step 6 stays on Gemini if the free tier covers it.

## Not planned

- An agent or ReAct loop. The pipeline is a fixed workflow: one search, then one answer.
