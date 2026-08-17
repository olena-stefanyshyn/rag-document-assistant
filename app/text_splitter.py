from pathlib import Path

from document_loader import load_pdf

def split_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks =[]
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            split_position = text.rfind("\n", start, end)

            if split_position == -1 or split_position <= start:
                split_position == text.rfind(". ", start, end)

            if split_position == -1 or split_position <= start:
                split_position = text.rfind(" ", start, end)

            if split_position > start:
                end = split_position + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - chunk_overlap, start + 1)

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

    print(f"Min chunk: {min(len(c) for c in chunks)}")
    print(f"Max chunk: {max(len(c) for c in chunks)}")
    print(f"Average chunk: {sum(len(c) for c in chunks) / len(chunks):.1f}")