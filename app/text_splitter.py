from pathlib import Path

from document_loader import load_pdf

def split_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
) -> list[str]:
    chunks =[]

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "data" / "raw" / "rag.pdf"

    text = load_pdf(path)
    chunks = split_text(text)

print(f"Document lenght: {len(text)}")
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk length: {len(chunks[0])}")
print(f"Last chunk length: {len(chunks[-1])}")