from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def create_reranker():
    """
    Create and return the cross-encoder reranker.
    """
    return CrossEncoder(RERANKER_MODEL_NAME)


def retrieve_and_rerank(
    query: str,
    vector_store,
    reranker,
    retrieval_k: int = 12,
    final_k: int = 5,
):
    """
    Retrieve candidate chunks from the vector store
    and rerank them using a cross-encoder.
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": retrieval_k,
        },
    )

    results = retriever.invoke(query)

    print("\nFAISS CANDIDATES:")

    for i, doc in enumerate(results, start=1):
        print(
            f"{i}. "
            f"document={doc.metadata.get('document')} | "
            f"page={doc.metadata.get('page')}"
        )

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

    print("\nAFTER RERANKING:")

    for i, (doc, score) in enumerate(
        reranked_results,
        start=1,
    ):
        print(
            f"{i}. "
            f"score={float(score):.4f} | "
            f"document={doc.metadata.get('document')} | "
            f"page={doc.metadata.get('page')}"
        )

    return reranked_results[:final_k]


def select_diverse_results(
    reranked_results,
    final_k: int = 3,
):
    """
    Select top reranked chunks while preserving
    document diversity when possible.
    """

    selected = []
    used_documents = set()

    # First pass:
    # take the best chunk from each document
    for doc, score in reranked_results:
        document = doc.metadata.get(
            "document",
            "unknown",
        )

        if document not in used_documents:
            selected.append(
                (doc, score)
            )

            used_documents.add(
                document
            )

        if len(selected) >= final_k:
            return selected

    # Second pass:
    # fill remaining slots with highest-ranked chunks
    for doc, score in reranked_results:

        if (doc, score) in selected:
            continue

        selected.append(
            (doc, score)
        )

        if len(selected) >= final_k:
            break

    return selected