import os
import torch
from transformers import pipeline

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    device=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    clean_up_tokenization_spaces=False
)

context = "Hugging Face is a company based in New York City that develops open-source NLP and ML tools."
question = "Where is Hugging Face based?"

messages = [
    {
        "role": "user",
        "content": f"Answer the question based only on the context provided.\n\nContext: {context}\nQuestion: {question}\nAnswer:"
    }
]

output = pipe(messages, max_new_tokens=40, do_sample=False)
print("Answer:", output[0]["generated_text"][-1]["content"].strip())