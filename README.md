# Knowledge Desk

**A document-based AI assistant that lets you upload files and ask questions about their contents.**

Knowledge Desk is a lightweight Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **TF-IDF search**, and the **Hugging Face Inference API**. It provides a simple browser interface for uploading PDF, TXT, and Markdown documents, then asking questions grounded in the uploaded material.

> **Important:** The app sends retrieved document text and your question to the configured Hugging Face model when you ask a question. Do not upload confidential or sensitive documents unless you are comfortable with that processing.

## Features

- Upload PDF, TXT, and Markdown (`.md`) documents.
- Extract text and split it into overlapping chunks.
- Retrieve relevant chunks using TF-IDF weighting and cosine similarity.
- Generate context-based answers with a Hugging Face chat model.
- Show source document names alongside answers.
- Persist the local index between server restarts.
- View, remove, or clear indexed documents.
- Use the built-in browser UI or FastAPI's interactive API docs.
- Configure the model and retrieval settings through environment variables.

## How it works

1. **Upload:** Text is extracted from a supported document.
2. **Chunk:** The text is split into overlapping sections.
3. **Index:** Each chunk is tokenized and stored in a local JSON index.
4. **Retrieve:** TF-IDF similarity selects the most relevant chunks for a question.
5. **Answer:** The selected context is sent to the configured Hugging Face chat model to produce an answer.

```text
Document → Text extraction → Chunks → TF-IDF index
                                         ↑
Question → TF-IDF retrieval → Relevant context → Hugging Face model → Answer + sources
```

This is a deliberately straightforward RAG implementation. TF-IDF is lightweight and easy to inspect, but it primarily matches words rather than deeper semantic meaning.

## Tech stack

- Python
- FastAPI and Uvicorn
- Hugging Face Inference API
- TF-IDF retrieval with cosine similarity
- pypdf for PDF text extraction
- HTML, CSS, and JavaScript frontend

## Project structure

```text
knowledge-desk/
├── backend/
│   ├── main.py          # FastAPI routes and app
│   ├── config.py        # Environment-based configuration
│   ├── ingest.py        # Document extraction and chunking
│   ├── rag.py           # Retrieval-augmented answer generation
│   └── tfidf_store.py   # TF-IDF index and persistence
├── frontend/
│   └── index.html       # Browser interface
├── data/
│   ├── uploads/         # Uploaded files (not committed)
│   └── index/           # Local index files (not committed)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer recommended
- A Hugging Face account and access to a compatible chat model for AI-generated answers
- Internet access when calling the Hugging Face API

## Run locally

### 1. Download or clone the repository

If you cloned it:

```bash
cd knowledge-desk
```

### 2. Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Hugging Face (optional in the app, needed for model answers)

Copy `.env.example` to `.env` and add your Hugging Face token.

**PowerShell**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux**
```bash
cp .env.example .env
```

Open `.env` and replace the placeholder `HF_API_KEY` with your own token. Never commit `.env` or share your token publicly. You can create a token in [Hugging Face Settings](https://huggingface.co/settings/tokens).

The browser interface may also provide a field for entering a token for a session. Treat tokens as secrets and use only your own token.

### 5. Start the server

Run this from the repository root:

```bash
uvicorn backend.main:app --reload
```

### 6. Open Knowledge Desk

- **App:** http://127.0.0.1:8000
- **Interactive API documentation:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/api/health

Upload a supported document, then ask a question about its contents.

## Configuration

The application reads these optional values from `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `HF_API_KEY` | Empty | Hugging Face API token |
| `HF_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Chat model ID |
| `CHUNK_SIZE` | `800` | Approximate characters per text chunk |
| `CHUNK_OVERLAP` | `120` | Characters shared between adjacent chunks |
| `TOP_K` | `4` | Number of relevant chunks retrieved |

The default model may require account access or model-provider availability. If it is unavailable to your account, configure a compatible model supported by the Hugging Face router.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Upload and index a PDF, TXT, or MD file |
| `POST` | `/api/ask` | Ask a question using the indexed documents |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{doc_id}` | Remove one indexed document |
| `DELETE` | `/api/documents` | Clear the knowledge base |
| `GET` | `/api/health` | Check application and index status |

## Limitations

- Retrieval is keyword-oriented; it may miss relevant passages phrased differently from the question.
- Scanned PDFs without an embedded text layer may not yield extractable text; OCR is not currently included.
- Generated answers can still be incorrect. Verify important information against the source documents.
- The project is intended for local development and demonstration, not an internet-facing production deployment. Add authentication, upload limits, request controls, and deployment hardening before exposing it publicly.
- Uploaded documents and the local index are stored on the machine running the app.

## Future improvements

- Add semantic retrieval with local embeddings.
- Add OCR support for scanned documents.
- Add authentication and upload-size limits.
- Add tests and continuous integration.
- Add streaming responses and richer source previews.
- Add Docker support and deployment instructions.

## Contributing

Issues and pull requests are welcome. For substantial changes, open an issue first to discuss the proposed improvement.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
