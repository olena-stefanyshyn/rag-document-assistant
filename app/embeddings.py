from pathlib import Path

from sentence_transformers import SentenceTransformer

from document_loader import load_pdf
from text_splitter import split_text

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks: list[str]):
    embeddings = model.encode(chunks)

    return embeddings

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "data" / "raw" / "rag.pdf"

    text = load_pdf(path)
    chunks = split_text(text)
    embeddings = create_embeddings(chunks)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")