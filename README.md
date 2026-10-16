# PAGESUTRA - AN AI POWERED CHATBOT

Ask questions across your PDFs and follow the citations back to the source page.
PAGESUTRA uses a React frontend, a FastAPI backend, and NVIDIA models for embeddings
and answers. It supports text PDFs and scanned pages through Tesseract OCR.

## Running locally

You need Python 3.11+, Node.js 20+, and an API key from [NVIDIA Build](https://build.nvidia.com/).
The default database configuration also needs Docker with Compose.

Start PostgreSQL from the repository root:

```sh
docker compose up -d postgres
```

Set up the API in PowerShell:

```powershell
cd backend
Copy-Item .env.example .env
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Set NVIDIA_API_KEY in .env before starting the API.
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

On macOS or Linux, use `cp .env.example .env`, `python3 -m venv .venv`,
and `source .venv/bin/activate` for the equivalent setup.

In another terminal, start the frontend:

```sh
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>, create an account, and upload a PDF.
The API docs are at <http://127.0.0.1:8001/docs>. Vite forwards `/api` requests
to that port. On Windows, use `npm.cmd` if PowerShell blocks `npm.ps1`.

For a simpler setup without Docker, leave `DATABASE_URL=` empty in `backend/.env`.
That uses SQLite and ranks vectors in Python. PostgreSQL uses pgvector for retrieval.
Uploaded files remain in `data/uploads/` with either database.

## Configuration

Copy `backend/.env.example` and adjust the values you need. Restart the API after changes.

| Setting | Purpose |
| --- | --- |
| `NVIDIA_API_KEY` | Required for uploads and answers |
| `NVIDIA_CHAT_MODEL`, `NVIDIA_EMBEDDING_MODEL` | Model identifiers for the NVIDIA endpoint |
| `EMBEDDING_DIMENSIONS` | PostgreSQL vector size; must match the embedding model |
| `DATABASE_URL` | PostgreSQL connection string, or empty for SQLite |
| `DATA_DIR` | Storage directory, relative to the project root unless absolute |
| `MAX_PDF_MB` | Upload limit per PDF; defaults to 20 MB |
| `OCR_ENABLED` | Enable OCR for image-bearing pages with little text |
| `TESSDATA_PREFIX` | Directory containing `eng.traineddata`; absolute or relative to `backend/` |
| `FRONTEND_ORIGIN` | Allowed origin for direct browser requests to the API |
| `COOKIE_SECURE` | Set to `true` when serving over HTTPS |

To read scanned PDFs, install Tesseract language data and point `TESSDATA_PREFIX`
to its `tessdata` directory. Set `OCR_ENABLED=false` if you only need text PDFs.
If you change the embedding model, delete and upload existing PDFs again: vectors
from different models are incompatible.

The Compose database credentials are for local development. Document text and
questions are sent to the configured NVIDIA API. Accounts use salted scrypt
password hashes and HttpOnly session cookies; documents and chats belong to the
signed-in account.

## Repository layout

```text
backend/
  app/
    main.py         # API routes and request handling
    auth.py         # Accounts and cookie sessions
    config.py       # Environment settings and storage paths
    db.py           # SQLite and PostgreSQL storage
    pdf.py          # Extraction, OCR, links, and page-aware chunks
    rag.py          # Embeddings, retrieval, and answer generation
  tests/
  .env.example
  migrate_sqlite_to_postgres.py
  requirements.txt
  requirements-dev.txt
frontend/
  src/
    components/     # Sign-in form, document library, and citations
    api.js          # Shared HTTP client
    App.jsx         # Workspace, document library, and chat
    main.jsx        # React entry point
    styles.css
compose.yaml
```

Local credentials, uploaded documents, databases, build output, presentation exports,
and scratch tools are excluded through `.gitignore`.

## How retrieval works

PyMuPDF extracts each page's text and external link annotations. Image-bearing pages
with fewer than 40 text characters use OCR when enabled. The text is split into
chunks of up to 1,200 characters with overlap, keeping each chunk on its source page.

The backend embeds those chunks with NVIDIA. Retrieval combines semantic cosine
similarity with BM25-style keyword matching and reciprocal rank fusion. Exact
names, identifiers, dates, and amounts can therefore compete with semantic matches.
Duplicate passages are removed, and a diversity penalty reduces repeated evidence.
PostgreSQL combines the top vector candidates with full-text keyword candidates;
SQLite ranks the account's selected chunks in Python.

Follow-up questions are rewritten into standalone queries using recent history.
A LangGraph search/generate/validate loop starts with six passages and retries
once with twelve if the answer lacks usable citations. LangChain handles prompts,
model calls, and output parsing. Answers include numbered references that open the
original PDF page. A draft with invalid/missing citations gets one model correction
attempt against the original evidence before it is rejected.

## Prompt design

Trusted task instructions are maintained in `backend/app/prompts.py` under a
`PROMPT_VERSION`. Separate prompts cover PDF answers, citation correction, follow-up
query rewriting, related background, research tool selection, evidence notes,
summaries, field extraction, and evidence review. The PDF answer prompt includes
an example of the required numeric citation format and an exact abstention phrase.
Question text, document passages, prior messages, and rejected drafts are encoded
as data in the human message; they never become system instructions. The prompts
tell the model to ignore instructions embedded in those data. Server-side citation,
ownership, and approval checks remain in code.

Evaluation JSON includes `prompt_version`, so a prompt edit can be checked against
the same test set before use. The seven synthetic cases include missing evidence
and instructions embedded in document text and conversation history. Prompt wording
helps guide the model but does not replace cited-page review.

The chat composer offers two answer modes. **PDF evidence only** is the default.
**PDFs + related background** adds a separately labeled explanation from model
knowledge. Only the document portion receives PDF citations. Background is not
verified against the PDF or live web, and must not fill in missing document-specific
facts such as a budget, date, identity, or contact detail. Neither mode can guarantee
a correct answer to every question; missing evidence is reported explicitly.

Citations are generated by the model and can be wrong. Answers with missing or
out-of-range source markers are replaced with an explicit abstention; the server
never adds a source marker on the model's behalf. No-context questions skip the
chat provider. A valid source number does not prove that the source supports a
claim, and the check does not detect every uncited claim in a mixed answer.
OCR also depends on scan quality. Uploads run within the request, so large scanned files can take a while;
there is no background job queue or upload progress yet.

## AI automation engineering skills demonstrated

| Job-description skill | Project evidence |
| --- | --- |
| Python and REST APIs | FastAPI upload, chat, document, account, and history endpoints; NVIDIA API integration |
| SQL | Parameterized SQLite/PostgreSQL queries, ownership checks, and migration script |
| Git workflow | GitHub Actions checks for backend tests, frontend formatting, and builds |
| LLM fundamentals and prompt design | Embeddings, context retrieval, system instructions, bounded conversation history, and citation guardrails |
| RAG and vector databases | Page-aware ingestion, OCR, overlapping chunks, pgvector cosine retrieval, and source links |
| LangChain document processing and ETL | Sequential passage-to-notes chains, summary generation, validated field extraction, and JSON reports |
| LangGraph and tool-calling agents | Researcher/writer/reviewer roles, read-only document tools, bounded cycles, durable checkpoints, and human approval |
| Evaluation sets | Seven synthetic answer-generation cases and a command that produces a scored JSON report |
| Logging and monitoring | JSON request events with request ID, route template, HTTP status, and duration |
| Auth and privacy | Account-owned documents and conversations; request telemetry omits questions, PDF text, cookies, and query strings |

Business-system integrations, live web research, and a verified cloud deployment
remain future extensions. The approved action currently saves a private report.
The current model provider is NVIDIA, accessed through an OpenAI-compatible chat
client; this is not an OpenAI- or Anthropic-hosted model integration.

## Document workflows

Open **Document workflows** in the sidebar, choose PDFs, select a workflow, and
enter a goal. You can reopen a run from **Saved runs** after a refresh or restart.

| Workflow | Processing |
| --- | --- |
| Summarize documents | Read selected passages, produce evidence notes in batches of four, then reduce them into a cited summary |
| Extract structured data | Read/map passages, extract named fields to a Pydantic-validated structure, validate source numbers, and export approved records as JSON |
| Research PDFs | A model selects `search_documents` or `read_document_page`; a writer drafts a report, and an evidence reviewer can request one revision |

Summary and extraction currently accept at most 24 chunks per run. Select fewer
or smaller documents if that limit is exceeded, or use research for a focused
question on larger documents. Research is limited to three tool rounds, two
executed tool calls per round, and 24 evidence passages. Extra model-requested
calls receive empty results. Research searches uploaded PDFs, not the internet.
All tools enforce the authenticated owner's selected document scope, independently
of model-generated arguments. No tools execute arbitrary SQL, code, or external actions.

Every workflow uses an evidence reviewer. One failed review returns the draft to
the writer; a second failure blocks saving. A successful review reaches a real
LangGraph `interrupt()`. The draft is checkpointed while the user reviews it.
**Approve and save** resumes with `Command(resume=True)` and saves the private
report; **Reject** finishes without saving one. Approval does not rerun completed
generation steps. The save operation is idempotent per workflow. Evidence review
is model-assisted and can be wrong: inspect the original cited pages yourself.
The implementation follows the official [LangGraph interrupt/resume pattern](https://docs.langchain.com/oss/python/langgraph/interrupts).

Workflow/report metadata uses the existing SQLite or PostgreSQL database.
Graph checkpoints, including source excerpts and drafts, use
`data/workflow-checkpoints.sqlite` (under `DATA_DIR` when customized). Back up both
databases together. Checkpoint storage is local to one host; multi-host deployment
needs a shared durable checkpointer. Requests run synchronously, not as background
jobs. Each completed graph step is checkpointed; **Retry from checkpoint** resumes
a failed step. A crash inside a step may repeat that step and its model calls.
An execution lease prevents overlapping runs and expires after ten minutes without
progress, allowing recovery of an abandoned request. Refresh or check progress to
see the current status. API errors preserve failed runs for recovery.

Deleting a PDF does not remove already saved workflow excerpts or reports. Use
**Delete run** to remove its report and checkpoint history too. Running workflows
must finish or be recovered before deletion. Document ownership is checked again
when saving a report.

Authenticated endpoints: `POST /api/workflows`, `GET /api/workflows`,
`GET /api/workflows/{id}`, `POST /api/workflows/{id}/decision` with a strict boolean
`approved`, `POST /api/workflows/{id}/retry`, `GET /api/workflows/{id}/report`, and
`DELETE /api/workflows/{id}`. Chat accepts `answer_mode` as `documents` or `background`.

## Answer-quality evaluations

From `backend/`, with a configured NVIDIA key, run:

```sh
python evaluate.py
```

This sends the synthetic questions and contexts in `evals/knowledge_agent.json`
to the configured chat model and writes `output/evals/report.json` at the repository
root. It uses provider credits and exits with code 1 if any case fails. Reports
include answers, citations, timing, and per-case checks. To use your own dataset
or output location, pass `--dataset path/to/cases.json --output path/to/report.json`.
Custom datasets are sent to NVIDIA too, and their answers are stored in the report.

Cases cover a supported date, multi-page evidence, an exact URL, an unknown fact,
empty context, document prompt injection, and conversation-history prompt injection.
Scoring checks expected phrases, citation numbers/pages, forbidden phrases, and
explicit abstention. These are heuristic regression checks, not a semantic quality
score. They exercise answer generation with fixed contexts; they do not measure
retrieval recall or prove resistance to all prompt injections. Inspect failed
answers and their source text before changing prompts or expectations.

To test real embeddings and hybrid retrieval as well as generation:

```sh
python evaluate.py --with-retrieval --output ../output/evals/retrieval-report.json
```

This creates an isolated temporary SQLite database with synthetic sources and
distractor passages, then reports relevant-page recall and answer checks. It does
not read or modify uploaded user documents. The small dataset is a regression
baseline, not a broad real-world quality benchmark.

For a live test of tool selection, summary, extraction, checkpoint restart,
approval, and background answers, run `python tests/live_workflow_smoke.py`.
This also uses only temporary synthetic documents and calls the configured NVIDIA
API. It tests SQLite checkpointing, not a production deployment.

The ordinary unit-test suite checks the evaluator, guardrails, and monitoring
without provider calls. A passing unit suite is separate from a passing live eval.

## Request monitoring

When running Uvicorn, each request emits an `http_request` JSON event to the
`uvicorn.error` logger. Responses include `X-Request-ID` for troubleshooting.
Route templates keep document and conversation IDs out of these events. Failed
requests record their HTTP status; unexpected exceptions record status 500.
These events support latency/error-rate monitoring through your log collector;
the project does not yet provide a monitoring dashboard or durable metrics store.

## Development

Run the API tests from `backend/`:

```sh
pip install -r requirements-dev.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

The tests use temporary SQLite databases and mock the NVIDIA calls. They cover
accounts, document ownership, PDF extraction, uploads, citations, deletion,
answer guardrails, hybrid retrieval, follow-ups, evaluation scoring, request
telemetry privacy, workflow ownership, approval/rejection, structured extraction,
checkpoint recovery, and deletion.
The OCR test skips when local Tesseract language data is unavailable.

Check the frontend from `frontend/`:

```sh
npm run format:check
npm run build
```

Use `npm run format` to format frontend source files.

For a live integration check, start the API with PostgreSQL and NVIDIA access,
create a test account, set `SMOKE_TEST_EMAIL` and `SMOKE_TEST_PASSWORD`, then run
`python tests/live_postgres_smoke.py` from `backend/`. It uploads a synthetic PDF,
checks a cited answer, and removes the document and chat afterwards. This calls
the external API. Override `API_BASE_URL` if the API runs elsewhere.

If moving an existing SQLite installation to PostgreSQL, stop the API, back up `data/`, set
`DATABASE_URL`, and run `python migrate_sqlite_to_postgres.py` from `backend/`.
Keep the uploaded PDF files alongside the migrated database. On older installations
with unowned records, the first account created receives those records.
The migration also copies workflow metadata and saved reports. Preserve the
checkpoint SQLite file in `DATA_DIR` so paused runs can still resume.
