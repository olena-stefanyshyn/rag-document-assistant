from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    prompt = f"""
    Answer the question using only the context below.
    Give a concise answer in one or two complete sentences.
    If the answer cannot be found in the context, say "I don't know."

    Question:
    {query}

    Context:
    {context}

    Answer:
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return answer

if __name__ == "__main__":
    query = "What is retrieval augmented generation?"

    context_chunks = [
        "Retrieval-augmented generation combines a pretrained parametric "
        "language model with non-parametric external memory.",

        "The retriever finds relevant documents and the generator uses "
        "those documents as additional context."
    ]

    answer = generate_answer(query, context_chunks)

    print(answer)