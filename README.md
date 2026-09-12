# contract-rag

Ask questions about legal contracts and get answers that cite the clause — and,
where the format has pages, the page — they stand on. Upload your own PDF, DOCX
or TXT and it becomes answerable immediately.

A citation that cannot be checked is the failure this project is built to avoid,
so an answer whose references do not match the excerpts it was given is rejected
rather than shown.

---

## How it works

```
upload:   bytes → load_document → chunk_by_clause → embed_and_index → contract_chunks
question: message + history → reformulate_query → retrieve(k=5)
                                                      ↓
                                        expand_with_related_clauses
                                                      ↓
                                               generate_answer → answer + citations
```

| Stage | Module | What it does |
| --- | --- | --- |
| Loader | `src/contract_rag/loader/` | PDF (page-aware), DOCX, TXT |
| Chunker | `src/contract_rag/chunker/` | Splits on clause boundaries; strategies for dotted, verbose and heading-only numbering |
| Embeddings | `src/contract_rag/embeddings/` | `ai-forever/ru-en-RoSBERTa`, upsert into Supabase pgvector |
| Retrieval | `src/contract_rag/retriever/` | Cosine top-k via the `match_documents` RPC, scoped to chosen documents |
| Generation | `src/contract_rag/generator/` | Structured, cited answers through LiteLLM |
| Ingestion | `src/contract_rag/ingestion/` | Load → chunk → index, for one uploaded file |
| HTTP | `api/` | `POST /chat`, `POST /documents` |
| UI | `frontend/` | Next.js + shadcn/ui chat workspace |

---

## Prerequisites

| Tool | Version | Notes |
| --- | --- | --- |
| Python | 3.12+ | `.python-version` pins 3.12 |
| [uv](https://docs.astral.sh/uv/) | latest | dependency and venv management |
| Node.js | 20+ | for the Next.js frontend |
| [Supabase](https://supabase.com/) project | — | Postgres with `pgvector`; the free tier is enough |
| [Supabase CLI](https://supabase.com/docs/guides/local-development/cli/getting-started) | latest | applies the migrations |
| [Ollama](https://ollama.com/) | latest | only if you use a local generation model |

The first question loads `ru-en-RoSBERTa` (about 1.5 GB) from Hugging Face and
caches it locally. The API loads it at start-up so the wait does not land on a
user's first question.

---

## Local setup

### 1. Install dependencies

```bash
uv sync
```

```bash
npm install --prefix frontend
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Fill in `.env`:

| Variable | Required | Meaning |
| --- | --- | --- |
| `SUPABASE_URL` | yes | Dashboard → Project Settings → API |
| `SUPABASE_KEY` | yes | the same page; the service role key if you want to write |
| `GENERATION_MODEL` | yes | LiteLLM model string, e.g. `ollama/qwen2.5:7b` |
| `JUDGE_MODEL` | only for `eval/` | the judge that scores answers; must differ from `GENERATION_MODEL` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | only for that provider | not needed with Ollama |

There is deliberately no default generation model: silently falling back to a
local model that may not be running turns a configuration mistake into a
timeout.

The frontend calls `http://localhost:8000` unless `NEXT_PUBLIC_API_URL` says
otherwise.

### 3. Create the database schema

Link the project once, then push the migrations in `supabase/migrations/`:

```bash
supabase link --project-ref <your-project-ref>
```

```bash
supabase db push
```

This creates the `contract_chunks` table, the `match_documents` search function
and the vector extension. Having a migration in the repository and having it
applied are different things — check with `supabase migration list --linked`.

### 4. Start the generation model

With Ollama, pull the model once:

```bash
ollama pull qwen2.5:7b
```

Ollama serves a model with a 4096-token window by default whatever the model
supports; the API asks for a larger one explicitly, so no configuration is
needed here.

To use a hosted provider instead, change `GENERATION_MODEL` and add its key —
no code changes.

---

## Running it

Backend (port 8000):

```bash
uv run uvicorn api.main:app --reload --port 8000
```

Frontend (port 3000):

```bash
npm run dev --prefix frontend
```

Open http://localhost:3000, drop a contract into the sidebar, and ask. Indexing
runs synchronously and takes seconds to minutes depending on the document.

---

## Using it

**A chat is about the documents uploaded into it.** Uploading scopes that
conversation to that file, and the header shows which one; clearing the scope
searches every indexed document. A chat that never uploaded anything searches
all of them.

**Every answer carries its excerpts.** Each citation badge opens into the
contract's own words, and "what search found" lists everything that reached the
model, with similarity scores — so a thin answer can be explained without
reading the server log.

**Follow-up questions are condensed first.** "And for how many days?" is
rewritten into a standalone question before retrieval; the line above the
citations shows the rewritten query whenever it differs from what you typed.

**DOCX and TXT have no pages,** so their citations name a clause and say "без
страницы" instead of inventing a page number.

---

## API

| Endpoint | Body | Returns |
| --- | --- | --- |
| `POST /chat` | `{message, history, source_files}` | `{answer, citations, standalone_query, excerpts}` |
| `POST /documents` | multipart `file` | `{filename, chunks_indexed, warnings}` |

`source_files` is optional; an empty list searches the whole index. Unsupported
formats, unreadable files and numbering the chunker does not recognise answer
with 4xx and a message meant for a person, not a stack trace.

---

## Tests

```bash
uv run pytest
```

No test touches the network: providers, the embedding model and Supabase are all
faked.

```bash
uv run ruff check api src tests
```

```bash
npm run lint --prefix frontend
```

### Evaluation

The harness in `eval/` scores retrieval and answer faithfulness over a hand-
labelled question set. It hits the live index and calls a model twice per
question, so it is kept out of `tests/` and run by hand:

```bash
uv run python eval/run_eval.py
```

`JUDGE_MODEL` must differ from `GENERATION_MODEL`: a model asked to grade its
own answers prefers them.

---

## Known limitations

- **Repeated clause numbers.** An annex that restarts numbering produces a
  second `5.1`; the stored rows are distinguished by a suffix, but a citation
  shows only "п. 5.1".
- **Small local models drift.** `qwen2.5:7b` can switch language mid-answer over
  a mixed-language corpus. The question's language is named explicitly in the
  prompt, which holds at the start of an answer but is not a guarantee.
- **Uploads are processed synchronously.** A large document blocks its request
  for as long as it takes to index.
- **A partial index is possible** if the connection drops between upsert batches
  of a document larger than 50 chunks.
- **DOCX content controls are skipped.** The loader walks direct children of the
  document body, so text inside `w:sdt`, text boxes or headers is not read.
