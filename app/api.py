from pathlib import Path
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.embeddings import create_vector_store
from app.retriever import create_reranker, retrieve_and_rerank
from app.generator import generate_answer


app = FastAPI(
    title="RAG Document Assistant",
    version="1.1.0",
)


# Global state

vector_store = None
reranker = None
current_document_name = None


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Request / response models

class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    page: int


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    vectors: int
    message: str


# Build vector store

def build_vector_store(pdf_path: Path):
    """
    Load PDF, split into chunks,
    create embeddings and FAISS vector store.
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

    store = create_vector_store(
        chunks
    )

    return (
        store,
        len(documents),
        len(chunks),
    )


# Startup

@app.on_event("startup")
def startup_event():
    global reranker

    print("Loading reranker...")

    reranker = create_reranker()

    print("RAG API is ready.")


# Root

@app.get("/")
def root():
    return {
        "message": "RAG Document Assistant API",
        "status": "running",
        "document": current_document_name,
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
    global current_document_name

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File must have a filename.",
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    file_path = (
        UPLOAD_DIR
        / file.filename
    )

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
        raise HTTPException(
            status_code=500,
            detail=f"Could not save file: {exc}",
        )

    try:
        (
            new_vector_store,
            pages,
            chunks,
        ) = build_vector_store(
            file_path
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not process PDF: "
                f"{exc}"
            ),
        )

    vector_store = new_vector_store
    current_document_name = file.filename

    return UploadResponse(
        filename=file.filename,
        pages=pages,
        chunks=chunks,
        vectors=vector_store.index.ntotal,
        message=(
            "Document uploaded and indexed successfully."
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
        raise HTTPException(
            status_code=409,
            detail=(
                "No document has been indexed. "
                "Upload a PDF first using /upload."
            ),
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

    if not context_chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                "No relevant context was found "
                "for the question."
            ),
        )

    answer = generate_answer(
        query=question,
        context_chunks=context_chunks,
    )

    sources = []
    seen_pages = set()

    for doc, score in reranked_results[:3]:
        page = doc.metadata.get(
            "page"
        )

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