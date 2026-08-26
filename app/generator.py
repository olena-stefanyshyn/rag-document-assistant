import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.eval()


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """
    Generate an answer using only the retrieved context.
    """

    context = "\n\n---\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a document question-answering assistant. "
                "Answer using only the provided context. "
                "Do not use outside knowledge. "

                "First identify the passage that most directly answers the question. "
                "Base your answer primarily on that passage. "

                "Prefer explicit statements from the context over related concepts "
                "or your own interpretation. "
                "Do not replace the mechanism stated in the context with another "
                "related mechanism. "

                "Preserve important technical terminology exactly when it is needed "
                "to answer the question. "
                "Preserve formulas, mathematical notation, quantities, and negation. "

                "If the context says that something is done 'by', 'through', "
                "'using', or 'by replacing' something, preserve that relationship "
                "in the answer. "

                "Do not reverse yes/no statements. "
                "If the context explicitly says something can be done, "
                "do not say that it cannot be done, and vice versa. "

                "Give a direct answer in 1-3 concise sentences. "

                "If the context does not contain enough information to answer, say: "
                "\"I don't have enough information in the provided context.\""
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer the question using the explicit evidence in the context."
            ),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[:, inputs.input_ids.shape[1]:]

    answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return answer.strip()