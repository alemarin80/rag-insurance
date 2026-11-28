# Retrieval improvement plan

Evaluation comes first. Each step changes one thing and logs one result.

| Step | Change | Needs new embeddings |
| :---: | :--- | :--- |
| 0 | Retrieval-only evaluation ([spec](specs/step0-retrieval-eval.md)). Done: `vector` and `hybrid` scored | Queries, once |
| 1 | BM25 alone, using `bm25s` with a Norwegian stemmer ([spec](specs/step1-bm25.md)). Done: level with `vector` on reworded questions, misses different ones | No |
| 2 | Sentence chunking (spaCy Norwegian), chunks allowed to cross pages | All chunks |
| 3 | Exact duplicate chunks stored once, with every source page | All chunks |
| 4 | Hybrid with Reciprocal Rank Fusion (BM25 and dense) ([spec](specs/step4-hybrid-bm25.md)). Done: `hybrid_bm25`, ahead of `hybrid` on both wordings, measurable only on the original | No |
| 5 | Reranker on the top 20 | No |
| 6 | Generation evaluation, with about 15 hand-graded answers | No |
| End | Score the `test` questions once | Queries, once |

## Rules

- Each step gets a spec, approval, code, a run and a result in the README with its flaws.
- The retrieval questions are generated once and committed. Later steps reuse them.
- Each question stores its answer sentence. A chunk is correct if it contains it, so the labels survive re-chunking.
- A change to the scorer gets a new version number. Compare numbers only within one version.
- Report chunk sizes with every chunking change: larger chunks contain a sentence more easily.
- Tune on the `dev` questions. Use the `test` questions for final numbers.
- Each logged run stores its Git commit.
- Don't run the ingest or `reset chunks` while the embedding API is unavailable. The ingest deletes all chunks before it calls the API.

## Decisions

- Embeddings stay on Gemini, using the free tier. The free tier can use the data you send to improve Google products.
- The model stays `gemini-embedding-2-preview`. Query embedding works with the current API key. The pricing page lists only `gemini-embedding-2`, and changing the model means re-embedding all chunks.
- The judge for step 6 stays on Gemini if the free tier covers it.

## Not planned

- An agent or ReAct loop. The pipeline is a fixed workflow: one search, then one answer.
