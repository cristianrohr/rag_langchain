# RAG PDF QA Pipeline (FAISS + LangChain)

A compact, production-minded Retrieval-Augmented Generation (RAG) pipeline that:

- Ingests PDFs from `documents/` using LangChain community loaders.
- Chunks and embeds text (OpenAI embeddings by default).
- Builds and persists a FAISS vector index (`faiss_index.index`).
- Retrieves top-k relevant chunks for a query.
- Uses an LLM to synthesize a final answer from retrieved context.

This repo is intentionally minimal while still showing end-to-end RAG mechanics you’ll use on real projects: ingestion, chunking, vector indexing, retrieval, LLM orchestration, and environment configuration.


## Features

- PDF ingestion via `PyPDFLoader`.
- Simple, transparent chunking for fast baseline retrieval.
- Embeddings via `OpenAIEmbeddings` (configurable model).
- FAISS index creation and on-disk persistence.
- Query → retrieve → generate answer using `ChatOpenAI`.
- `.env`-driven configuration (API keys, knobs) with `python-dotenv`.
- Practical notes on deprecation warnings and macOS OpenMP quirks.


## Architecture

Data flow at a glance:

```
PDFs (documents/) ──> Loader ──> Chunker ──> Embeddings ──> FAISS Index (disk)
                                     ▲                             │
                                     │                             ▼
                               Query (text) ──> Embed ──> Top‑K search ──> LLM answer
```

Key modules:
- `src/rag.py`: RAG building blocks (config, embeddings, FAISS, retrieval, LLM wrapper)
- `src/pipeline.py`: Simple orchestration entry for a user query


## Repo Structure

```
.
├── documents/                 # Place your PDFs here
├── src/
│   ├── pipeline.py            # Orchestrates end-to-end query → answer
│   └── rag.py                 # RAGConfig, LLMConfig, RAG implementation
├── requirements.txt
├── .env                       # Local config (not for committing)
└── faiss_index.index          # Generated FAISS index (created on first run)
```


## Getting Started

### Prerequisites
- Python 3.10+
- An OpenAI API key (set `OPENAI_API_KEY` in `.env`)
- Either `uv` (recommended) or `pip` + `venv`

Install `uv`: https://docs.astral.sh/uv/

### Setup

1) Clone and enter the project folder:

```
git clone <your-fork-url>
cd rag
```

2) Add PDFs to `documents/`.

3) Create your environment and install dependencies:

- Using `uv` (recommended):
```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

- Using `pip` + `venv`:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4) Configure environment variables:

- Copy `.env` and set your OpenAI key (do not commit secrets):
```
OPENAI_API_KEY=sk-...
```

### Run

```
uv run python src/pipeline.py
# or
python src/pipeline.py
```

On first run, the pipeline will:
- Load PDFs, chunk text, embed with the configured model.
- Build a FAISS index and persist it to `faiss_index.index`.
- Answer the sample query in `src/pipeline.py`.


## Configuration

Defaults are defined in code and can be overridden by editing the constructors:

- `RAGConfig(embedding_model="text-embedding-3-small", faiss_index_path="faiss_index.index")`
- `LLMConfig(model_name="gpt-4.1-nano", temperature=0.7)`

A `.env` file is loaded automatically via `dotenv`. The sample `.env` includes:

- `OPENAI_API_KEY`: needed for embeddings + LLM.
- Optional knobs like `TOP_K`, `INDEX_PATH`, etc. (The scaffolding is in place; you can wire these directly into `RAGConfig` if desired.)


## Suppressing LangChain Warnings (Optional)

Add this at the top of your entrypoint (before importing LangChain) to reduce deprecation noise:

```python
import warnings, logging
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
for n in ("langchain", "langchain_core", "langchain_community"):
    logging.getLogger(n).setLevel(logging.ERROR)
```

Or via CLI:
```
PYTHONWARNINGS="ignore::DeprecationWarning" python src/pipeline.py
```


## Notes on macOS + OpenMP (FAISS)

On some macOS setups, you may see an OpenMP duplicate runtime error (`libomp.dylib`). This project sets `KMP_DUPLICATE_LIB_OK=TRUE` at import time as a pragmatic workaround. For production-grade setups, prefer one of:
- Use a consistent toolchain (e.g., Conda) so only a single OpenMP runtime is present.
- Reinstall `faiss-cpu`/`numpy` to ensure ABI compatibility.
- Consider alternative vector stores (e.g., HNSW) if OpenMP conflicts persist.


## Migration: `langchain-openai` (Recommended)

LangChain has moved OpenAI integrations into a separate package. To remove deprecation warnings and follow current best practices:

```
pip install -U langchain-openai
```

Then update imports in `src/rag.py`:

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
```

And keep using the standard LangChain `invoke` API:

```python
response = llm.invoke("your prompt")
print(response.content)
```


## Troubleshooting

- Changed embedding model but index exists? Delete `faiss_index.index` and re-run to rebuild with the new dimension.
- Rate limits / auth errors: verify `OPENAI_API_KEY` and organization settings.
- No answers returned: confirm PDFs are present and non-empty; increase `top_k` or chunk size; try a larger embedding model.


## Roadmap / Ideas

- Replace naive chunking with `RecursiveCharacterTextSplitter` and metadata-based retrieval.
- Add reranking (e.g., cross-encoder) and/or rephrasing (HyDE) for tougher queries.
- Swap FAISS for a managed vector DB (Chroma, Pinecone, Weaviate) behind a common interface.
- Stream responses and add a lightweight UI (Gradio/Streamlit).
- Add automated evaluation (answer faithfulness, context coverage) and unit tests.


## Security

- Never commit secrets. Keep `.env` local and out of version control.
- Rotate keys if a secret is accidentally exposed.


## License

This repository is intended for portfolio and educational use. Add a license if you plan to distribute or build upon it publicly.

