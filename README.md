# RAG Document Assistant

A multi-document Retrieval-Augmented Generation (RAG) system for question answering over PDF documents.

The system retrieves relevant passages using FAISS, reranks them with a cross-encoder, and generates answers with a local Qwen model. Answers include document and page-level sources.

## Architecture

```text
PDF Documents
      ↓
Text Chunking
      ↓
SentenceTransformer Embeddings
      ↓
FAISS Vector Search
      ↓
Cross-Encoder Reranking
      ↓
Diversity-Aware Context Selection
      ↓
Qwen LLM
      ↓
Answer + Sources
```

## Tech Stack

- Python 3.11
- LangChain
- FAISS
- Sentence Transformers
- Hugging Face Transformers
- PyTorch
- FastAPI
- Docker / Docker Compose
- GitHub Actions
- Oracle Cloud

### Models

- Embeddings: `sentence-transformers/all-mpnet-base-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generator: `Qwen/Qwen2.5-0.5B-Instruct`

## Features

- Multi-document PDF question answering
- Persistent FAISS vector index
- Two-stage retrieval with cross-encoder reranking
- Diversity-aware context selection
- Document and page-level source attribution
- Local LLM generation
- FastAPI REST API with Swagger UI
- Dockerized cloud deployment
- GitHub Actions CI

## Evaluation

| Metric | Result |
|---|---:|
| Hit@1 | 80% |
| Hit@3 | 100% |
| MRR | 0.90 |
| Retrieval Accuracy | 100% |
| Document Coverage | 100% |
| Answer Keyword Accuracy | 40% |

Retrieval performance is strong, while answer generation remains the main bottleneck.

The relatively low Answer Keyword Accuracy is partly caused by strict keyword-based evaluation and by the lightweight `Qwen2.5-0.5B-Instruct` model used for CPU-friendly inference.

Possible improvements include:
- using a stronger generation model;
- semantic answer evaluation instead of strict keyword matching;
- prompt and context-selection optimization;
- evaluation on a larger test set.


## Run with Docker

```bash
git clone https://github.com/olena-stefanyshyn/rag-document-assistant.git
cd rag-document-assistant
docker compose up --build
```

Swagger UI:

```text
http://localhost:8000/docs
```

## CI & Deployment

GitHub Actions automatically verifies the project on pushes and pull requests by installing dependencies and building the Docker image.

The application is deployed on an Oracle Cloud Ubuntu ARM64 VM using Docker Compose.

```text
GitHub
   ↓
Oracle Cloud VM
   ↓
Docker Compose
   ↓
FastAPI
   ↓
RAG Pipeline
```

## Future Improvements

- Stronger generation model
- Semantic answer evaluation
- Structured logging and monitoring
- HTTPS / reverse proxy
- Automated deployment (CD)
