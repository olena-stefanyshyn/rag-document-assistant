import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

model.eval()


def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a document question-answering assistant. "
                "Answer the question using only the provided context. "
                "Read all relevant passages before answering. "
                "Give a direct and complete answer in 2-4 concise sentences. "
                "Preserve technical terminology, formulas, and mathematical notation exactly. "
                "Do not change the meaning of mathematical symbols or quantities. "
                "Distinguish vector representations from scores, probabilities, and model outputs. "
                "Ignore context that is unrelated to the question. "
                "If the context does not contain enough information to answer, say: "
                "\"I don't have enough information in the provided context.\""
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Explain the answer directly and completely."
            )
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]

    answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return answer.strip()