from pathlib import Path
import shutil
import logging
import time

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.embeddings import (
    create_embedding_model,
    create_vector_store,
)

from app.retriever import (
    create_reranker,
    retrieve_and_rerank,
    select_diverse_results,
)

from app.generator import generate_answer


# Logging

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# FastAPI app

app = FastAPI(
    title="RAG Document Assistant",
    version="1.3.0",
)


# Global state

vector_store = None
reranker = None
indexed_documents = []


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "uploads"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "faiss_index"
)

DOCUMENTS_FILE = (
    INDEX_DIR
    / "documents.txt"
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Request / response models

class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    document: str
    page: int


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    vectors: int
    indexed_documents: list[str]
    message: str


# Build vector store for one PDF

def build_vector_store(pdf_path: Path):
    """
    Load one PDF, split it into chunks,
    attach document metadata,
    and build a temporary FAISS vector store.
    """

    loader = PyPDFLoader(
        str(pdf_path)
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(
        documents
    )

    # Add source filename to every chunk
    for chunk in chunks:
        chunk.metadata["document"] = (
            pdf_path.name
        )

    store = create_vector_store(
        chunks
    )

    return (
        store,
        len(documents),
        len(chunks),
    )


# Save vector store

def save_vector_store():
    """
    Save the complete FAISS index
    and list of indexed documents.
    """

    if vector_store is None:
        return

    vector_store.save_local(
        str(INDEX_DIR)
    )

    DOCUMENTS_FILE.write_text(
        "\n".join(indexed_documents),
        encoding="utf-8",
    )


# Load vector store

def load_vector_store():
    """
    Load previously saved FAISS index.
    """

    index_file = (
        INDEX_DIR
        / "index.faiss"
    )

    metadata_file = (
        INDEX_DIR
        / "index.pkl"
    )

    if not index_file.exists():
        return None

    if not metadata_file.exists():
        return None

    embeddings = create_embedding_model()

    store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return store


# Load indexed document names

def load_indexed_documents():
    """
    Load document names stored alongside the index.
    """

    if not DOCUMENTS_FILE.exists():
        return []

    content = DOCUMENTS_FILE.read_text(
        encoding="utf-8"
    )

    documents = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    return documents


# Startup

@app.on_event("startup")
def startup_event():
    global vector_store
    global reranker
    global indexed_documents

    logger.info("Loading reranker...")

    reranker = create_reranker()

    logger.info(
        "Checking for saved FAISS index..."
    )

    try:
        saved_store = load_vector_store()

        if saved_store is not None:
            vector_store = saved_store

            indexed_documents = (
                load_indexed_documents()
            )

            logger.info(
                "Saved FAISS index loaded."
            )

            logger.info(
                "Indexed documents: %s",
                indexed_documents,
            )

        else:
            logger.info(
                "No saved FAISS index found."
            )

    except Exception:
        logger.exception(
            "Could not load saved FAISS index."
        )

    logger.info("RAG API is ready.")


# Root

@app.get("/")
def root():
    return {
        "message": (
            "RAG Document Assistant API"
        ),
        "status": "running",
        "index_loaded": (
            vector_store is not None
        ),
        "documents": indexed_documents,
    }


# Upload PDF

@app.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...)
):
    global vector_store
    global indexed_documents

    start_time = time.perf_counter()

    logger.info(
        "Document upload started | filename=%s",
        file.filename,
    )

    if not file.filename:
        logger.warning(
            "Upload rejected | missing filename"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "File must have a filename."
            ),
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):
        logger.warning(
            "Upload rejected | unsupported file | filename=%s",
            file.filename,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files are supported."
            ),
        )

    if file.filename in indexed_documents:
        logger.warning(
            "Upload rejected | already indexed | filename=%s",
            file.filename,
        )

        raise HTTPException(
            status_code=409,
            detail=(
                "This document has already "
                "been indexed."
            ),
        )

    file_path = (
        UPLOAD_DIR
        / file.filename
    )

    # Save uploaded PDF
    try:
        with open(
            file_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

    except Exception as exc:
        logger.exception(
            "Could not save uploaded file | filename=%s",
            file.filename,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save file: "
                f"{exc}"
            ),
        )

    logger.info(
        "Uploaded file saved | filename=%s",
        file.filename,
    )

    # Build temporary vector store
    try:
        (
            new_vector_store,
            pages,
            chunks,
        ) = build_vector_store(
            file_path
        )

    except Exception as exc:
        logger.exception(
            "Could not process PDF | filename=%s",
            file.filename,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not process PDF: "
                f"{exc}"
            ),
        )

    logger.info(
        "PDF processed | filename=%s pages=%d chunks=%d",
        file.filename,
        pages,
        chunks,
    )

    # Merge new index into existing one
    try:
        if vector_store is None:
            vector_store = (
                new_vector_store
            )

        else:
            vector_store.merge_from(
                new_vector_store
            )

        indexed_documents.append(
            file.filename
        )

        save_vector_store()

    except Exception as exc:
        logger.exception(
            "Could not merge or save FAISS index | filename=%s",
            file.filename,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not merge or save "
                f"FAISS index: {exc}"
            ),
        )

    logger.info(
        (
            "Document indexed successfully | "
            "filename=%s pages=%d chunks=%d "
            "vectors=%d duration=%.2fs"
        ),
        file.filename,
        pages,
        chunks,
        vector_store.index.ntotal,
        time.perf_counter() - start_time,
    )

    return UploadResponse(
        filename=file.filename,
        pages=pages,
        chunks=chunks,
        vectors=(
            vector_store.index.ntotal
        ),
        indexed_documents=(
            indexed_documents
        ),
        message=(
            "Document uploaded, indexed, "
            "merged, and saved successfully."
        ),
    )


# Ask question

@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask_question(
    request: QuestionRequest,
):
    if vector_store is None:
        logger.warning(
            "Question rejected | no vector index"
        )

        raise HTTPException(
            status_code=409,
            detail=(
                "No documents have been indexed. "
                "Upload a PDF first using /upload."
            ),
        )

    if reranker is None:
        logger.error(
            "Question rejected | reranker unavailable"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Reranker is not ready."
            ),
        )

    question = (
        request.question.strip()
    )

    if not question:
        logger.warning(
            "Question rejected | empty question"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Question cannot be empty."
            ),
        )

    start_time = time.perf_counter()

    logger.info(
        "Question received | length=%d",
        len(question),
    )

    try:
        reranked_results = retrieve_and_rerank(
            query=question,
            vector_store=vector_store,
            reranker=reranker,
            retrieval_k=12,
            final_k=12,
        )

    except Exception:
        logger.exception(
            "Retrieval or reranking failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Retrieval or reranking failed."
            ),
        )

    logger.info(
        "Retrieval and reranking completed | candidates=%d",
        len(reranked_results),
    )

    selected_results = select_diverse_results(
        reranked_results,
        final_k=3,
    )

    logger.info(
        "Context selected | chunks=%d",
        len(selected_results),
    )

    context_chunks = [
        doc.page_content
        for doc, score
        in selected_results
    ]

    if not context_chunks:
        logger.warning(
            "No relevant context found"
        )

        raise HTTPException(
            status_code=404,
            detail=(
                "No relevant context was "
                "found for the question."
            ),
        )

    generation_start = time.perf_counter()

    logger.info(
        "Generation started | context_chunks=%d",
        len(context_chunks),
    )

    try:
        answer = generate_answer(
            query=question,
            context_chunks=context_chunks,
        )

    except Exception:
        logger.exception(
            "Answer generation failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Answer generation failed."
            ),
        )

    generation_duration = (
        time.perf_counter()
        - generation_start
    )

    logger.info(
        "Generation completed | duration=%.2fs",
        generation_duration,
    )

    sources = []
    seen_sources = set()

    for doc, score in selected_results:
        page = doc.metadata.get(
            "page"
        )

        document = doc.metadata.get(
            "document",
            "unknown",
        )

        if page is None:
            continue

        human_page = page + 1

        source_key = (
            document,
            human_page,
        )

        if source_key not in seen_sources:
            sources.append(
                Source(
                    document=document,
                    page=human_page,
                )
            )

            seen_sources.add(
                source_key
            )

    total_duration = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        (
            "Question answered successfully | "
            "sources=%d generation=%.2fs "
            "total_duration=%.2fs"
        ),
        len(sources),
        generation_duration,
        total_duration,
    )

    return AnswerResponse(
        answer=answer,
        sources=sources,
    )