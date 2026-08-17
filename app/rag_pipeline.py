from pathlib import Path

from document_loader import load_pdf
from text_splitter import split_text
from embeddings import create_embeddings
from retriever import retrieve_top_k
from generator import generate_answer

def run_rag(query: str, pdf_path: Path, k: int = 3) -> str:
    text = load_pdf(pdf_path)

    chunks = split_text(text)

    chunk_embeddings = create_embeddings(chunks)

    query_embedding = create_embeddings([query])[0]

    results = retrieve_top_k(
        query_embedding,
        chunk_embeddings,
        chunks,
        k=k,
    )

    for result in results:
        print("SCORE:", result["score"])
        print("CHUNK:")
        print(result["chunk"])
        print("-" * 80)

    context_chunks = [
        result["chunk"]
        for result in results
    ]

    answer = generate_answer(
        query,
        context_chunks,
    )

    return answer


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "raw" / "rag.pdf"

    query = "What is retrieval augmented generation?"

    answer = run_rag(
        query=query,
        pdf_path = pdf_path,
        k=3,
    )

    print(answer)