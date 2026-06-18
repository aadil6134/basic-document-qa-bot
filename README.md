# Basic Document Q&A Bot

A Python Retrieval-Augmented Generation (RAG) app that answers questions from local PDF, DOCX, and TXT documents. It indexes source documents into a persistent ChromaDB vector store, retrieves the most relevant chunks for each question, and asks Gemini to answer only from that retrieved context with inline citations.

## Features

- Extracts text from PDF, DOCX, and TXT files.
- Preserves source metadata such as filename, page number, and chunk range.
- Uses overlapping recursive-style chunks for better semantic retrieval.
- Persists embeddings locally in `db/` with ChromaDB.
- Uses Gemini embeddings and Gemini generation through `google-generativeai`.
- Provides both an interactive CLI and a Streamlit UI.
- Refuses to answer when retrieved documents do not contain the answer.

## Project Structure

```text
document-qa-bot/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── sample_company_handbook.txt
├── db/
└── src/
    ├── __init__.py
    ├── config.py
    ├── ingest.py
    ├── query.py
    └── main.py
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file:

```bash
copy .env.example .env
```

4. Add your Gemini API key to `.env`:

```text
GEMINI_API_KEY=your_real_key
```

5. Put your source files in `data/`. Supported formats are `.pdf`, `.docx`, and `.txt`.

## Usage

Index documents:

```bash
python -m src.ingest --data-dir data --db-path db --reset
```

Ask a single question:

```bash
python -m src.query "What is the vacation policy?"
```

Start the interactive CLI:

```bash
python -m src.main
```

Run the Streamlit UI:

```bash
streamlit run src/main.py
```

## How It Works

1. `src/ingest.py` scans `data/`, extracts text page by page, splits it into overlapping chunks, embeds each chunk with Gemini, and stores the vectors in ChromaDB.
2. `src/query.py` embeds the user's question with the same embedding model, retrieves the top matching chunks, builds a strict grounded prompt, and asks Gemini for an answer.
3. Each context block is labeled with a source citation, such as `(sample_company_handbook.txt, Page 1)`.
4. If the retrieved context does not contain an answer, the prompt instructs the model to say so instead of guessing.

## Configuration

Configuration is read from `.env` and defaults are defined in `src/config.py`.

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Required API key for Gemini embeddings and generation. |
| `GEMINI_GENERATION_MODEL` | Gemini text generation model. |
| `GEMINI_EMBEDDING_MODEL` | Gemini embedding model. |
| `CHROMA_COLLECTION` | ChromaDB collection name. |

## Notes For Submission

The assignment asks for:

- A public GitHub repository with the project source and README.
- A published or deployed project link.
- A 3 to 5 minute screen recording walking through the project and code.

For deployment, Streamlit Community Cloud is a simple fit. Add `GEMINI_API_KEY` as a Streamlit secret instead of committing it to the repository.
