from pathlib import Path

from pypdf import PdfReader

def load_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def load_documents(directory: Path) -> dict[str, str]:
    documents = {}

    for file_path in directory.glob("*.pdf"):
        documents[file_path.name] = load_pdf(file_path)

    return documents


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"

    documents = load_documents(data_dir)

    for name, text in documents.items():
        print(f"{name}: {len(text)} characters")

    print(text[:2000])