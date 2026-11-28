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
   python -m src.ops.data status
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

## Build the data

Run these commands from the repository root with the virtual environment active.

### Build

After you change the PDFs, the chunk size or `EMBED_MODEL`, run:

```bash
python -m src.ingestion.ingest               # replace all chunks
python -m src.ops.data reset eval_set --yes  # old questions are stale
python -m src.evals.run_eval                 # new questions and benchmark
```

### Check

To see row counts, chunks per PDF and whether the eval set exists, run:

```bash
python -m src.ops.data status
```

### Reset

`reset` only deletes and asks for confirmation (`--yes` skips it). Choose a command:

| Command | Deletes |
| :--- | :--- |
| `python -m src.ops.data reset chunks` | All chunks |
| `python -m src.ops.data reset eval_logs` | All eval runs |
| `python -m src.ops.data reset user_feedback` | Simulated feedback |
| `python -m src.ops.data reset user_feedback --include-real` | All feedback |
| `python -m src.ops.data reset eval_set` | `data/eval_set.json` |
| `python -m src.ops.data reset chunks eval_logs eval_set` | All three |

### Start over

After you change `EMBED_DIM`, run these commands, then [Build](#build). They delete the whole database, including real feedback:

```bash
docker compose down -v
docker compose up -d
```

## Test your setup

Run the unit tests. They don't call any LLM API, but the code creates a Gemini client when it loads, so `GEMINI_API_KEY` must be set. Any value works for the unit tests. GitHub Actions runs the same command on every push (`.github/workflows/ci.yml`):

```bash
GEMINI_API_KEY=dummy python -m pytest -v
```

To check that each provider and model works, run the live smoke test. It costs a few cents. Add `--e2e` to include one full RAG question with judging:

```bash
python -m tests.smoke_llm --e2e
```

## Improving retrieval

The goal is the process: change one thing, measure it, and write down what the measurement can't tell. Evaluation flaws are fixed in a new scorer version, and numbers are compared only within one version. The `test` questions stay unscored until the final comparison. For the order of work, see the [plan](docs/plan.md).

A retrieved chunk is correct if it contains the question's answer sentence, whichever document it comes from (scorer version 2, see the [spec](docs/specs/step0-retrieval-eval.md)).

### Results

The following table shows scorer version 2 on 53 `dev` questions:

| Method | Hit@1 | Hit@5 | Hit@20 | MRR@20 | nDCG@5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `vector` | 0.623 | 0.830 | 0.962 | 0.707 | 0.726 |
| `hybrid` | 0.641 | 0.830 | 0.962 | 0.721 | 0.737 |

`hybrid` scored better on 1 question and worse on none. The difference on nDCG@5 is +0.011 (95% interval 0.000 to +0.032), which is no measurable difference.

### What changed since the pilot

The [pilot](docs/reports/step0-results.md) marked results by page and showed `hybrid` ahead. Part of that lead came from identical chunks that tie and appear in arbitrary order. Pages that answered a question could also be marked wrong. Version 2 marks by text, so identical chunks always get the same mark. It also drops questions made from the same text and questions their source text can't answer.

### Known flaws

- Some questions don't name the insurance, and policies give different amounts. Only the source policy's sentence counts. For example, `q064` and `q088` ask about the same theft limit, and the correct amount differs between two home policies.
- The same sentence with a small difference in wording counts as wrong.
- An AI assistant picked the answer sentences and checked most `test` questions. The `dev` questions aren't hand-checked.
- The questions come from the policy text, so real questions are probably harder.
- With 53 questions, only large differences are measurable.
- Source chunks were picked with a fixed seed, evenly across the 5 PDFs, but without a minimum length. Some were page tails that answer nothing; 4 such questions were dropped.

### Run it

| Command | Does |
| :--- | :--- |
| `python -m src.evals.retrieval_eval generate` | Writes the question file, once (paid LLM calls). New questions need an `answer` field before scoring |
| `python -m src.evals.retrieval_eval check` | Hand-checks the `test` questions |
| `python -m src.evals.retrieval_eval run --mode vector` | Scores `vector` on `dev` and logs the run |
| `python -m src.evals.retrieval_eval run --mode hybrid --baseline RUN_ID` | Scores and compares with an earlier run |

Replace `RUN_ID` with the ID that an earlier run printed. `run` needs committed code.

## Old end-to-end evaluation

`python -m src.evals.run_eval` scores whole answers, not search alone. A simulator LLM wrote 18 questions in `data/eval_set.json`: 15 from randomly drawn chunks and 3 trap questions that the policies don't answer. For each retrieval mode, the generator answers from the top 5 chunks, and a judge LLM scores faithfulness and relevance. Results, latency and tokens go to the `eval_logs` table. Step 6 of the [plan](docs/plan.md) replaces it.

| Mode | Recall@5 | Faithfulness | Relevance | Average latency |
| :--- | :---: | :---: | :---: | :---: |
| `vector` | 0.80 | 0.99 | 0.98 | 15,988 ms |
| `hybrid` (RRF) | 0.80 | 0.97 | 0.96 | 13,678 ms |

`gpt-5-mini` answered, `gemini-4-hydrogen` judged and `gpt-5-nano` wrote the questions. One question changes Recall@5 by 0.07, so the modes don't differ measurably. At `LLM_EFFORT=low` (now the default), `vector` took 7,542 ms per answer, with about 1,700 input and 550 output tokens. The `hybrid` re-run stopped with a 429 error: the Gemini judge hit its spending cap.

Its flaws:

- The chunks are drawn with `ORDER BY RANDOM()`: no seed, larger PDFs get more questions, and a page tail of a few characters can be drawn. The retrieval evaluation doesn't use this draw.
- Recall@5 counts only the exact source page, the flaw that scorer version 2 fixed.
- 15 answerable questions measure only very large differences.
- The judge has no reference answer, and its scores aren't checked against human grades.

## Next steps

- Follow the [retrieval improvement plan](docs/plan.md).
- Add a streaming endpoint. The answer is structured JSON (`answer`, `citations`, `is_covered_in_context`), so a half-written answer can't be shown as it arrives. One option is to send the retrieved sources first and the final JSON last. Real token streaming needs a prompt that writes the plain answer before the citations.
