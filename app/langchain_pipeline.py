from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from generator import generate_answer


def build_vector_store(pdf_path: Path):
    """
    Load a PDF, split it into chunks, create embeddings,
    and build a FAISS vector store.
    """

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"Pages: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    print(f"Vectors stored: {vector_store.index.ntotal}")

    return vector_store


def ask_question(
    query: str,
    vector_store,
    reranker,
):
    """
    Retrieve candidate chunks, rerank them,
    and generate an answer from the best context.
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 12,
        },
    )

    results = retriever.invoke(query)

    pairs = [
        (query, doc.page_content)
        for doc in results
    ]

    scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(results, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\nTOP RETRIEVED RESULTS:")

    for i, (doc, score) in enumerate(
        reranked_results[:5],
        start=1,
    ):
        print(f"\nResult {i}")
        print(f"Reranker score: {float(score):.4f}")
        print(f"Page: {doc.metadata.get('page')}")
        print(doc.page_content)

    context_chunks = [
        doc.page_content
        for doc, score in reranked_results[:3]
    ]

    answer = generate_answer(
        query=query,
        context_chunks=context_chunks,
    )

    sources = reranked_results[:3]

    return answer, sources


def print_sources(sources):
    """
    Print unique PDF pages used as context.
    """

    seen_pages = set()

    print("\nSOURCES:")

    for doc, score in sources:
        page = doc.metadata.get("page")

        if page is not None and page not in seen_pages:
            # LangChain page numbering starts from 0
            print(f"- Page {page + 1}")
            seen_pages.add(page)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    pdf_path = (
        project_root
        / "data"
        / "raw"
        / "rag.pdf"
    )

    print("Building vector store...")

    vector_store = build_vector_store(
        pdf_path=pdf_path
    )

    print("\nLoading reranker...")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    print("\nRAG assistant is ready.")

    while True:
        query = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        answer, sources = ask_question(
            query=query,
            vector_store=vector_store,
            reranker=reranker,
        )

        print("\nFINAL ANSWER:")
        print(answer)

        print_sources(sources)