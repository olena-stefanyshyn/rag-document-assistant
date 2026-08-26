from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import create_vector_store
from retriever import create_reranker, retrieve_and_rerank
from generator import generate_answer


TEST_CASES = [
    {
        "question": "What is non-parametric memory in RAG?",
        "reference_answer": (
            "Non-parametric memory in RAG is a dense vector index "
            "of Wikipedia accessed by a neural retriever."
        ),
        "expected_keyword_groups": [
            ["dense vector index", "vector index"],
            ["wikipedia"],
        ],
        "relevant_pages": [2],
    },
    {
        "question": "What is the difference between RAG-Token and RAG-Sequence?",
        "reference_answer": (
            "RAG-Sequence uses the same retrieved document to predict "
            "each target token, while RAG-Token can use a different "
            "document for each target token."
        ),
        "expected_keyword_groups": [
            ["rag-sequence"],
            ["rag-token"],
            ["same document"],
            ["different document"],
        ],
        "relevant_pages": [3],
    },
    {
        "question": "How does DPR work?",
        "reference_answer": (
            "DPR uses a bi-encoder architecture with separate document "
            "and query encoders. Documents and queries are represented "
            "as dense vectors and compared using inner product."
        ),
        "expected_keyword_groups": [
            ["bi-encoder", "bi encoder"],
            ["document encoder"],
            ["query encoder"],
        ],
        "relevant_pages": [3],
    },
    {
        "question": "Does retrieving more documents always improve performance?",
        "reference_answer": (
            "No. Retrieving more documents improves RAG-Sequence performance "
            "in the reported open-domain QA experiment, but RAG-Token performance "
            "peaks at around 10 retrieved documents."
        ),
        "expected_keyword_groups": [
            [
                "does not always",
                "not always",
                "doesn't always",
            ],
        ],
        "relevant_pages": [8],
    },
    {
        "question": "How can RAG's knowledge be updated at test time?",
        "reference_answer": (
            "RAG's knowledge can be updated at test time by replacing "
            "or hot-swapping its non-parametric memory or retrieval index."
        ),
        "expected_keyword_groups": [
            [
                "replace",
                "replacing",
                "hot-swap",
                "hot-swapping",
                "hot swapped",
            ],
            [
                "non-parametric memory",
                "retrieval index",
                "index",
            ],
        ],
        "relevant_pages": [7, 8],
    },
]


def build_vector_store(pdf_path: Path):
    """
    Load the PDF, split it into chunks,
    and build the FAISS vector store.
    """

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)

    return create_vector_store(chunks)


def get_page(doc):
    """
    Convert LangChain zero-based page number
    into a human-readable page number.
    """

    page = doc.metadata.get("page")

    if page is None:
        return None

    return page + 1


def get_retrieved_pages(reranked_results):
    """
    Return unique human-readable page numbers
    from retrieved results.
    """

    pages = []

    for doc, score in reranked_results:
        page = get_page(doc)

        if page is not None and page not in pages:
            pages.append(page)

    return pages


def is_relevant(doc, relevant_pages):
    """
    Check whether a retrieved document belongs
    to one of the manually labelled relevant pages.
    """

    page = get_page(doc)

    if page is None:
        return False

    return page in relevant_pages


def hit_at_k(
    reranked_results,
    relevant_pages,
    k,
):
    """
    Return 1 if at least one relevant result
    occurs within the top-k results.
    """

    top_k_results = reranked_results[:k]

    for doc, score in top_k_results:
        if is_relevant(doc, relevant_pages):
            return 1

    return 0


def reciprocal_rank(
    reranked_results,
    relevant_pages,
):
    """
    Return reciprocal rank of the first relevant result.

    rank 1 -> 1.0
    rank 2 -> 0.5
    rank 3 -> 0.333...
    """

    for rank, (doc, score) in enumerate(
        reranked_results,
        start=1,
    ):
        if is_relevant(doc, relevant_pages):
            return 1.0 / rank

    return 0.0


def check_answer(
    answer,
    expected_keyword_groups,
):
    """
    Each group contains alternative acceptable phrases.

    At least one phrase from every group
    must occur in the generated answer.
    """

    answer_lower = answer.lower()

    for group in expected_keyword_groups:

        group_match = any(
            keyword.lower() in answer_lower
            for keyword in group
        )

        if not group_match:
            return False

    return True


def evaluate():
    project_root = Path(__file__).resolve().parent.parent

    pdf_path = (
        project_root
        / "data"
        / "raw"
        / "rag.pdf"
    )

    print("Building vector store...")

    vector_store = build_vector_store(
        pdf_path
    )

    print("Loading reranker...")

    reranker = create_reranker()

    total = len(TEST_CASES)

    retrieval_correct = 0
    answer_correct = 0

    hit_1_total = 0
    hit_3_total = 0
    reciprocal_rank_total = 0.0

    for i, test in enumerate(
        TEST_CASES,
        start=1,
    ):
        question = test["question"]
        relevant_pages = test["relevant_pages"]

        print("\n" + "=" * 70)
        print(f"TEST {i}")
        print(f"Question: {question}")

        reranked_results = retrieve_and_rerank(
            query=question,
            vector_store=vector_store,
            reranker=reranker,
            retrieval_k=12,
            final_k=12,
        )

        hit_1 = hit_at_k(
            reranked_results=reranked_results,
            relevant_pages=relevant_pages,
            k=1,
        )

        hit_3 = hit_at_k(
            reranked_results=reranked_results,
            relevant_pages=relevant_pages,
            k=3,
        )

        rr = reciprocal_rank(
            reranked_results=reranked_results,
            relevant_pages=relevant_pages,
        )

        hit_1_total += hit_1
        hit_3_total += hit_3
        reciprocal_rank_total += rr

        retrieval_success = bool(hit_3)

        if retrieval_success:
            retrieval_correct += 1

        context_chunks = [
            doc.page_content
            for doc, score in reranked_results[:3]
        ]

        answer = generate_answer(
            query=question,
            context_chunks=context_chunks,
        )

        answer_success = check_answer(
            answer=answer,
            expected_keyword_groups=(
                test["expected_keyword_groups"]
            ),
        )

        if answer_success:
            answer_correct += 1

        retrieved_pages = get_retrieved_pages(
            reranked_results[:3]
        )

        print("\nReference answer:")
        print(test["reference_answer"])

        print("\nGenerated answer:")
        print(answer)

        print("\nRelevant pages:")
        print(relevant_pages)

        print("\nRetrieved top-3 pages:")
        print(retrieved_pages)

        print(
            "\nHit@1: "
            f"{'PASS' if hit_1 else 'FAIL'}"
        )

        print(
            "Hit@3: "
            f"{'PASS' if hit_3 else 'FAIL'}"
        )

        print(
            f"Reciprocal Rank: {rr:.4f}"
        )

        print(
            "Answer correctness: "
            f"{'PASS' if answer_success else 'FAIL'}"
        )

    hit_at_1_score = hit_1_total / total
    hit_at_3_score = hit_3_total / total
    mrr = reciprocal_rank_total / total

    retrieval_accuracy = (
        retrieval_correct / total
    )

    answer_accuracy = (
        answer_correct / total
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")

    print(
        f"Hit@1: "
        f"{hit_1_total}/{total} "
        f"({hit_at_1_score:.2%})"
    )

    print(
        f"Hit@3: "
        f"{hit_3_total}/{total} "
        f"({hit_at_3_score:.2%})"
    )

    print(
        f"MRR: {mrr:.4f}"
    )

    print(
        f"Retrieval accuracy: "
        f"{retrieval_correct}/{total} "
        f"({retrieval_accuracy:.2%})"
    )

    print(
        f"Answer accuracy: "
        f"{answer_correct}/{total} "
        f"({answer_accuracy:.2%})"
    )


if __name__ == "__main__":
    evaluate()