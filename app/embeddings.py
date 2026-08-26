from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def create_embedding_model():
    """
    Create and return the embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    return embeddings


def create_vector_store(chunks):
    """
    Create a FAISS vector store from LangChain Document chunks.
    """

    embeddings = create_embedding_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store