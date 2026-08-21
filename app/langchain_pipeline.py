from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from generator import generate_answer


# -----------------------------
# 1. Path to PDF
# -----------------------------

project_root = Path(__file__).resolve().parent.parent

pdf_path = (
    project_root
    / "data"
    / "raw"
    / "rag.pdf"
)


# -----------------------------
# 2. Load PDF
# -----------------------------

loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

print(f"Pages: {len(documents)}")


# -----------------------------
# 3. Split documents into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks: {len(chunks)}")


# -----------------------------
# 4. Embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# -----------------------------
# 5. Vector database
# -----------------------------

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print(f"Vectors stored: {vector_store.index.ntotal}")


# -----------------------------
# 6. User question
# -----------------------------

query = "Does retrieving more documents always improve RAG performance?"


# -----------------------------
# 7. Retriever
# -----------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 12,
    }
)


# -----------------------------
# 8. Initial retrieval
# -----------------------------

results = retriever.invoke(query)


# -----------------------------
# 9. Reranker
# -----------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

pairs = [
    (query, doc.page_content)
    for doc in results
]

scores = reranker.predict(pairs)


# -----------------------------
# 10. Sort results by relevance
# -----------------------------

reranked_results = sorted(
    zip(results, scores),
    key=lambda item: item[1],
    reverse=True
)


# -----------------------------
# 11. Print retrieved documents
# -----------------------------

for i, (doc, score) in enumerate(
    reranked_results,
    start=1
):
    print(f"\nResult {i}")
    print(f"Reranker score: {float(score):.4f}")
    print(f"Page: {doc.metadata.get('page')}")
    print(doc.page_content)


# -----------------------------
# 12. Select best chunks
# -----------------------------

context_chunks = [
    doc.page_content
    for doc, score in reranked_results[:3]
]


# -----------------------------
# 13. Generate final answer
# -----------------------------

answer = generate_answer(
    query=query,
    context_chunks=context_chunks,
)


# -----------------------------
# 14. Output
# -----------------------------

print("\nFINAL ANSWER:")
print(answer)