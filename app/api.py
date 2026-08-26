from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.embeddings import create_vector_store
from app.retriever import create_reranker, retrieve_and_rerank
from app.generator import generate_answer


app = FastAPI(
    title="RAG Document Assistant",
    version="1.0.0",
)


# Global objects

vector_store = None
reranker = None


# Request / response models

class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    page: int


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


# Build vector store

def build_vector_store(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(
        documents
    )

    vector_store = create_vector_store(
        chunks
    )

    return vector_store


# Startup

@app.on_event("startup")
def startup_event():
    global vector_store
    global reranker

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    pdf_path = (
        project_root
        / "data"
        / "raw"
        / "rag.pdf"
    )

    if not pdf_path.exists():
        raise RuntimeError(
            f"PDF file not found: {pdf_path}"
        )

    print("Building vector store...")

    vector_store = build_vector_store(
        pdf_path
    )

    print(
        f"Vectors stored: "
        f"{vector_store.index.ntotal}"
    )

    print("Loading reranker...")

    reranker = create_reranker()

    print("RAG API is ready.")


# Health endpoint

@app.get("/")
def root():
    return {
        "message": "RAG Document Assistant API",
        "status": "running",
    }


# Ask endpoint

@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask_question(
    request: QuestionRequest,
):
    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not ready.",
        )

    if reranker is None:
        raise HTTPException(
            status_code=503,
            detail="Reranker is not ready.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    reranked_results = retrieve_and_rerank(
        query=question,
        vector_store=vector_store,
        reranker=reranker,
        retrieval_k=12,
        final_k=5,
    )

    context_chunks = [
        doc.page_content
        for doc, score
        in reranked_results[:3]
    ]

    answer = generate_answer(
        query=question,
        context_chunks=context_chunks,
    )

    sources = []

    seen_pages = set()

    for doc, score in reranked_results[:3]:
        page = doc.metadata.get("page")

        if page is None:
            continue

        human_page = page + 1

        if human_page not in seen_pages:
            sources.append(
                Source(
                    page=human_page
                )
            )

            seen_pages.add(
                human_page
            )

    return AnswerResponse(
        answer=answer,
        sources=sources,
    )