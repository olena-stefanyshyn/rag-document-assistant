import numpy as np

from pathlib import Path

from document_loader import load_pdf
from text_splitter import split_text
from embeddings import create_embeddings

def cosine_similarity(query_vector, document_vectors):
    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)

    similarities = document_vectors @ query_vector
    similarities = similarities / (document_norms * query_norm)

    return similarities

def retrieve_top_k(
        query_embedding,
        chunk_embeddings,
        chunks,
        k: int = 3,
):
    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings,
    )

    top_indices = np.argsort(similarities)[-k:][::-1]

    results = []

    for index in top_indices:
        results.append(
            {
                "chunk": chunks[index],
                "score": float(similarities[index]),
            }
        )

    return results

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "data" / "raw" / "rag.pdf"

    text = load_pdf(path)
    chunks = split_text(text)
    chunk_embeddings = create_embeddings(chunks)

    query = "What model uses bidirectional transformers?"
    query_embedding = create_embeddings([query])[0]

    results = retrieve_top_k(
        query_embedding,
        chunk_embeddings,
        chunks,
        k=5,
    )

    for result in results:
        print(f"Score: {result["score"]:.3f}")
        print(result["chunk"])
        print("-" * 80)