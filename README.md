# Norwegian insurance policy RAG and evaluation

This repository shows how to build and evaluate a retrieval-augmented generation (RAG) pipeline over Norwegian insurance policy terms (*forsikringsvilkår*) without an orchestration framework. It is for engineers who want a small, readable example of hybrid retrieval, citation-grounded answers, and automated evaluation. After the Quickstart, you can ask policy questions through an API and score the answers with a simulated user and a judge based on a large language model (LLM).

This repository doesn't include a web UI, authentication, or the policy PDFs.

## Architecture

The pipeline has four parts:

- **Storage and retrieval** (`src/ingestion/`, `src/retrieval/`): PostgreSQL 16 with `pgvector` stores 768-dimension `gemini-embedding-2-preview` embeddings in an HNSW (Hierarchical Navigable Small World) index. A GIN (Generalized Inverted Index) over `to_tsvector('norwegian', ...)` provides keyword search. Reciprocal Rank Fusion (RRF) merges both result lists.
- **Grounded generation** (`src/generation/rag.py`): `ask()` retrieves the top chunks, then the configured LLM returns a Pydantic-validated answer with `[doc_name, page_number, quote]` citations. It refuses when the retrieved excerpts don't answer the question.
- **Automated evaluation** (`src/evals/run_eval.py`): A question simulator generates a fixed Norwegian question set in `data/eval_set.json`. The set covers exact clause lookups, informal customer scenarios, exclusion edge cases, and unanswerable trap questions. An LLM judge scores faithfulness and relevance, and every run is logged to the `eval_logs` table. A second script, `src/evals/simulate_users.py` (`python -m src.evals.simulate_users`), plays three simulated policyholders in two-turn conversations through `ask()` and logs their ratings to `user_feedback`.
- **API and feedback** (`src/api/app.py`): A FastAPI service with `/health`, `/ask`, and `/feedback`. `/feedback` stores thumbs-up or thumbs-down ratings in the `user_feedback` table. The `source` column marks each row as `real` (from the API) or `simulated` (from `simulate_users`).

![System flow. Top row: policy PDFs are ingested into Postgres, which the ask() pipeline searches to retrieve chunks and generate an answer. Blue row: simulation and evaluation scripts. Orange row: real use through the FastAPI service.](docs/flowchart.svg)

The top row is the pipeline. The blue row is the simulation and evaluation scripts and the tables they write. The orange row is real use through the API. The two `user_feedback` boxes are one table: the `source` column tells simulated ratings from real ones.

## Design choices

- **Chunking:** 1,200-character windows with 200-character overlap, cut by character count, so a chunk can start or end mid-sentence.
- **Hybrid retrieval:** The vector side and the keyword side each return 30 candidates. RRF scores each chunk as the sum of `1 / (60 + rank)` over both lists, and the top 5 go to the generator.
- **Keyword query:** The keyword side uses `plainto_tsquery`, which requires every query term to match. A long question can return few or no keyword rows, and then the ranking comes mostly from the vector side.
- **No reranker:** The pipeline passes the top 5 chunks straight to the generator.
- **Embeddings:** Always Gemini at 768 dimensions, so the vector index doesn't change when you switch the generator or the judge.

## Choose providers and models

The generator, the judge, and the simulator can each use Gemini, OpenAI, or Anthropic. Set these variables in `.env`:

| Variable | Purpose |
| :--- | :--- |
| `LLM_PROVIDER`, `LLM_MODEL` | Answer generation. If unset, the provider is `openai` and the model is the provider default in `src/llm.py`. |
| `JUDGE_PROVIDER`, `JUDGE_MODEL` | LLM judge that scores answers. |
| `SIM_PROVIDER`, `SIM_MODEL` | Simulator that writes the eval questions and plays the simulated users. |
| `LLM_EFFORT`, `JUDGE_EFFORT`, `SIM_EFFORT` | Reasoning effort per role: `minimal`, `low`, `medium`, `high`, or `default` (the provider's own default). The defaults are `low` for the generator and the simulator and `default` for the judge. |

If you leave a role's provider unset, it falls back to `LLM_PROVIDER`. The model falls back to `LLM_MODEL` only when the provider is the same, otherwise to the provider default in `src/llm.py`.

The docstring of `src/llm.py` lists suggested models and marks which ones the smoke test has run. Use a judge from a different provider than the generator to avoid self-preference bias. `run_eval` and `simulate_users` refuse to start when any two of the three roles (generator, judge, simulator) use the same provider and model. To allow it on purpose, set `ALLOW_SAME_MODELS=1`. Embeddings always use Gemini, so a Gemini API key is required in every setup.

The effort setting maps to `reasoning.effort` for OpenAI reasoning models (`gpt-5*`, `o*`) and to `thinking_level` for Gemini. Anthropic and `gpt-4*` models ignore it. A lower effort cuts latency and cost, and it can lower answer quality, so keep the judge at `default` unless you have measured the effect.

## Quickstart

Before you begin, make sure you have:

- Docker with Compose
- Python 3.10 or later
- A Gemini API key (embeddings), plus a key for each other provider you choose
- Norwegian insurance policy PDFs

To run the pipeline:

1. Start Postgres:

   ```bash
   docker compose up -d
   ```

1. Create a virtual environment and install the dependencies:

   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

1. Create a `.env` file in the repository root:

   ```bash
   DATABASE_URL=postgresql://rag:rag@localhost:5432/insurance_rag
   GEMINI_API_KEY=GEMINI_KEY
   OPENAI_API_KEY=OPENAI_KEY
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-5-mini
   JUDGE_PROVIDER=gemini
   JUDGE_MODEL=gemini-4-hydrogen
   SIM_PROVIDER=openai
   SIM_MODEL=gpt-5-nano
   ```

   Replace `GEMINI_KEY` and `OPENAI_KEY` with your API keys. Add `ANTHROPIC_API_KEY` if you use Anthropic.

1. Copy your PDF files into `data/policies/`, then ingest them:

   ```bash
   python -m src.ingestion.ingest
   ```

1. Confirm that the `chunks` table has rows:

   ```bash
   docker exec insurance_rag_db psql -U rag -d insurance_rag -c "SELECT count(*) FROM chunks;"
   ```

1. Run the benchmark. It compares `vector` and `hybrid` retrieval and prints a summary line per mode:

   ```bash
   python -m src.evals.run_eval
   ```

1. Optional: start the API. Skip this step if you only run evals, because `run_eval` calls the pipeline directly:

   ```bash
   uvicorn src.api.app:app --reload
   ```

1. Optional: confirm that the API is up. The response is `{"status":"ok"}`:

   ```bash
   curl http://localhost:8000/health
   ```

## Reset the data

Use these commands when you add or change PDFs, change the chunk size or the embedding model, or want fresh eval results. Run them from the repository root with the virtual environment active.

To rebuild the policy chunks, run the ingest again. It deletes every row in `chunks` first, so you can run it any time:

```bash
python -m src.ingestion.ingest
```

Confirm that the table has about one chunk per 1,000 characters of policy text:

```bash
docker exec insurance_rag_db psql -U rag -d insurance_rag -c "SELECT doc_name, count(*) FROM chunks GROUP BY doc_name ORDER BY 1;"
```

After you re-ingest, regenerate the eval questions. They come from random chunks, so old questions no longer match the new content. Delete the file, and the next `run_eval` writes a new `data/eval_set.json` (see [Question generation](#question-generation)):

```bash
rm data/eval_set.json
python -m src.evals.run_eval
```

To clear logged results without touching the policy chunks, truncate the log tables. This deletes every eval run and every feedback row, real and simulated:

```bash
docker exec insurance_rag_db psql -U rag -d insurance_rag -c "TRUNCATE eval_logs, user_feedback RESTART IDENTITY;"
```

To start over completely, delete the database volume. This removes `chunks`, `eval_logs`, and `user_feedback`. Then run the Quickstart from step 1 again:

```bash
docker compose down -v
docker compose up -d
python -m src.ingestion.ingest
```

If you change `EMBED_MODEL` or `EMBED_DIM` in `src/llm.py`, run `docker compose down -v` instead of the plain ingest, because the `chunks` table keeps its old vector size.

## Test your setup

Run the unit tests. They don't call any LLM API, but the code creates a Gemini client when it loads, so `GEMINI_API_KEY` must be set. Any value works for the unit tests. GitHub Actions runs the same command on every push (`.github/workflows/ci.yml`):

```bash
GEMINI_API_KEY=dummy python -m pytest -v
```

To check that each provider and model works, run the live smoke test. It costs a few cents. Add `--e2e` to include one full RAG question with judging:

```bash
python -m tests.smoke_llm --e2e
```

## How the evaluation works

The command `python -m src.evals.run_eval` runs three stages:

1. **Generate questions.** If `data/eval_set.json` doesn't exist, a simulator LLM writes a question set and saves it there. Later runs reuse the file, so every run and every model you try answers the same questions.
1. **Retrieve and answer.** For each question, the pipeline runs once per retrieval mode (`vector`, then `hybrid`). The mode only changes how `src/retrieval/search.py` picks the top 5 chunks. The generator LLM then answers from those chunks.
1. **Judge.** A separate judge LLM scores each answer. It receives the question, the answer, the answer's `is_covered_in_context` flag, and the retrieved chunks. It doesn't receive the retrieval mode or the source page of the question. It does know whether the question is a trap. The script writes each result to `eval_logs`, with the mode, the generator and judge model names and effort, the retrieved pages, whether the expected page was among them, and the generator's token counts. Output tokens include reasoning tokens.

### Question generation

The simulator writes 18 questions: 15 from policy text and 3 fixed trap questions. To write the 15, it does the following for each of three question types:

1. Pick 5 random chunks from the `chunks` table (`ORDER BY RANDOM()`).
1. Send each chunk and the instruction for that type to the simulator LLM (`SIM_PROVIDER`, `SIM_MODEL`, temperature 0.7 where the model accepts one: `gpt-5*` models ignore it). The LLM returns one Norwegian question as structured output.
1. Store the question with `expected_source`, the `document#page` of the chunk it came from.

The three draws are independent, so one chunk can produce questions of different types, and a PDF with more chunks tends to get more questions.

The following table lists the question types:

| Type | Count | Instruction to the simulator |
| :--- | :---: | :--- |
| `exact_lookup` | 5 | Ask for an exact number, age, day limit, amount, or clause condition, without copying the surrounding sentence. |
| `messy_user` | 5 | Write an informal, vague, real-life scenario from a stressed customer, without insurance jargon. |
| `exclusion_edge` | 5 | Ask whether a specific exception or limitation applies. |
| `trap` | 3 | Fixed, hand-written questions that the policies don't answer, such as crypto fraud or dog dental care. `expected_source` is `NONE`. |

### What the scores measure

The run prints three scores per retrieval mode:

- **Recall@5** (recall at 5, answerable questions only): 1 if the `expected_source` page is among the 5 retrieved chunks, otherwise 0. It measures retrieval alone, with no generator or judge model involved.
- **Faithfulness** (judge, 0 to 1): 1 if no claim goes beyond the retrieved chunks. The judge has no faithfulness rule for the 3 trap questions, but their scores are still in the average over all 18 questions.
- **Relevance** (judge, 0 to 1): 1 if the answer addresses the question. For a trap question, 1 only if the bot refused.

### Find the retrieval misses

`run_eval` prints each miss with the expected page and the pages it retrieved. The same data stays in `eval_logs`, so you can inspect a miss without a new run:

```bash
docker exec insurance_rag_db psql -U rag -d insurance_rag -c "SELECT question, expected_source, retrieved FROM eval_logs WHERE retrieval_hit = false ORDER BY id DESC;"
```

### Known weaknesses

The design has these limits:

- Each question comes from one chunk, so the set has no questions that need several policies or tables.
- Recall@5 matches the exact page. A question that another page answers equally well counts as a miss.
- The judge has no reference answer. It checks the answer against the retrieved chunks only, so an answer grounded in the wrong chunks can still score high.
- The scores are LLM estimates. This repository doesn't validate them against human grades.
- The simulator sees the chunk, so a question can echo its wording and make retrieval look easier than it is for real customers.

To change the question mix, edit `styles` and `n_per_type` in `generate_eval_set()` in `src/evals/run_eval.py`. Then delete `data/eval_set.json`, as described in [Reset the data](#reset-the-data).

## Benchmark results

The following table compares retrieval modes on 18 Norwegian questions: 15 written from the policy chunks and 3 unanswerable trap questions. The corpus is 5 policy PDFs, split into 209 chunks. `gpt-5-mini` generated the answers, `gemini-4-hydrogen` judged them, and `gpt-5-nano` wrote the questions.

| Mode | Recall@5 | Faithfulness | Relevance | Average latency |
| :--- | :---: | :---: | :---: | :---: |
| `vector` | 0.80 | 0.99 | 0.98 | 15,988 ms |
| `hybrid` (RRF) | 0.80 | 0.97 | 0.96 | 13,678 ms |

One question changes Recall@5 by 0.07 (1 of 15) and the judge scores by about 0.06 (1 of 18), so the two modes don't differ in any way this set can measure. The identical Recall@5 may have a cause besides equal quality: the keyword side of `hybrid` requires every query term to match (see [Design choices](#design-choices)), so on full questions `hybrid` may behave like `vector`. This repository hasn't tested that.

The table above used the default reasoning effort, where `gpt-5-mini` takes 14 to 16 seconds per answer. A new run of `vector` at `LLM_EFFORT=low` (now the default) gave Recall@5 0.80, faithfulness 1.00, relevance 0.99, an average latency of 7,542 ms, and about 1,700 input and 550 output generator tokens per answer. The `hybrid` re-run didn't finish: the Gemini judge returned a 429 error because the project hit its monthly spending cap. The three misses of that run are in `eval_logs`. Gemini with `thinking_level="low"` answered in 3 to 4 seconds in earlier runs on a smaller, partly ingested corpus, so treat that comparison as unverified.

Run `python -m src.evals.run_eval` to reproduce the table with your own `.env` settings. The numbers change when you regenerate the questions, because the simulator's output varies.

## Next steps

To make the comparison meaningful, work in this order:

1. Test the keyword query: replace `plainto_tsquery` with `websearch_to_tsquery` or an OR query, and re-run the benchmark.
2. Grow the eval set beyond 18 questions, with multi-policy and table-heavy questions.
3. Hand-grade about 15 answers and compare them with the judge's scores.
4. If Recall@5 stays at 0.80, try a larger `top_k` or add a reranker.
5. Add a streaming endpoint. The answer is structured JSON (`answer`, `citations`, `is_covered_in_context`), so a half-written answer can't be shown as it arrives. One option is to send the retrieved sources first and the final JSON last. Real token streaming needs a prompt that writes the plain answer before the citations.
