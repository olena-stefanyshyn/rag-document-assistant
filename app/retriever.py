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

    return reranked_results[:final_k]