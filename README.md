# Norwegian insurance policy RAG and evaluation

This repository shows how to build and evaluate a retrieval-augmented generation (RAG) pipeline over Norwegian insurance policy terms (*forsikringsvilkår*) without an orchestration framework. It is for engineers who want a small, readable example of hybrid retrieval, citation-grounded answers, and automated evaluation. After the Quickstart, you can ask policy questions through an API and score the answers with a simulated user and an LLM judge.

This repository doesn't include a web UI, authentication, or the policy PDFs.

## Architecture

The pipeline has four parts:

- **Storage and retrieval** (`src/ingestion/`, `src/retrieval/`): PostgreSQL 16 with `pgvector` stores 768-dimension `gemini-embedding-2-preview` embeddings in an HNSW (Hierarchical Navigable Small World) index. A GIN (Generalized Inverted Index) over `to_tsvector('norwegian', ...)` provides keyword search. Reciprocal Rank Fusion (RRF) merges both result lists.
- **Grounded generation** (`src/generation/rag.py`): The configured LLM returns a Pydantic-validated answer with `[doc_name, page_number, quote]` citations. It refuses when the retrieved excerpts don't answer the question.
- **Automated evaluation** (`src/evals/run_eval.py`): A user simulator generates a fixed Norwegian question set in `data/eval_set.json`. The set covers exact clause lookups, informal customer scenarios, exclusion edge cases, and unanswerable trap questions. An LLM judge scores faithfulness and relevance, and every run is logged to the `eval_logs` table.
- **API and feedback** (`src/api/app.py`): A FastAPI service with `/health`, `/ask`, and `/feedback`. `/feedback` stores thumbs-up or thumbs-down ratings in the `user_feedback` table.

```mermaid
flowchart LR
    PDF[Policy PDFs] -->|PyMuPDF + embeddings| DB[(Postgres: pgvector HNSW + Norwegian tsvector GIN)]
    Q[User / simulator] --> API[FastAPI /ask]
    API -->|Hybrid RRF| DB
    DB -->|Top-5 chunks| LLM[Configured LLM + Pydantic]
    LLM -->|Answer + citations| API
    API -->|Thumbs up/down| FB[(user_feedback)]
    LLM -->|Scored by| JUDGE[LLM judge] --> EVAL[(eval_logs)]
```

## Choose providers and models

The generator and the judge can each use Gemini, OpenAI, or Anthropic. Set these variables in `.env`:

| Variable | Purpose |
| :--- | :--- |
| `LLM_PROVIDER`, `LLM_MODEL` | Answer generation. If unset, the provider is `openai` and the model is the provider default in `src/llm.py`. |
| `JUDGE_PROVIDER`, `JUDGE_MODEL` | Question simulator, LLM judge, and simulated users. If unset, they fall back to the `LLM_*` values. |

`src/llm.py` lists suggested models and marks which ones the smoke test has run. Use a judge from a different provider than the generator to avoid self-preference bias. Embeddings always use Gemini, so a Gemini API key is required in every setup.

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

1. Start the API:

   ```bash
   uvicorn src.api.app:app --reload
   ```

1. Confirm that the API is up. The response is `{"status":"ok"}`:

   ```bash
   curl http://localhost:8000/health
   ```

## Test your setup

Run the unit tests. They don't call any LLM API:

```bash
python -m pytest -v
```

To check that each provider and model works, run the live smoke test. It costs a few cents. Add `--e2e` to include one full RAG question with judging:

```bash
python -m tests.smoke_llm --e2e
```

## Benchmark results

The following table compares retrieval modes on 18 Norwegian queries: 15 generated from the policies and 3 unanswerable trap questions. These numbers come from an earlier run with Gemini as both generator and judge. <!-- TODO(verify): rerun `python -m src.evals.run_eval` with the current .env and replace this table. -->

| Mode | Thinking level | Recall@5 | Faithfulness | Relevance | Average latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `vector` (baseline) | default | 1.00 | 1.00 | 1.00 | 11,882 ms |
| `vector` | `low` | 1.00 | 1.00 | 1.00 | 3,180 ms |
| `hybrid` (RRF) | `low` | 1.00 | 1.00 | 1.00 | 4,138 ms |

Setting `thinking_level="low"` on Gemini cut average latency by about 73% (11.9 s to 3.2 s) with no drop in faithfulness or refusal accuracy. This setting applies to Gemini only.

## Limitations

- All three modes score 1.00, so the eval set can't yet separate them. A harder set with multi-policy and table-heavy questions would show a difference.
- The retrieval step has no reranker.
- `python -m src.evals.simulate_users` writes to the `user_feedback` table, which the API creates on startup. Start the API once before you run it.
