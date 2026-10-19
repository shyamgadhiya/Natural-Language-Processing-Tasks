import torch
from transformers import pipeline

# device=0 if you have CUDA installed; otherwise remove device argument for CPU
device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    device=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

text_to_summarize = """
Hugging Face is an open-source data science and machine learning platform. 
It acts as a hub for AI experts and enthusiasts—like a GitHub for AI. 
It enables users to build, train, and deploy machine learning models based 
on open-source code and technologies.
"""

messages = [
    {
        "role": "user",
        "content": f"Summarize the following text concisely:\n\n{text_to_summarize}"
    }
]

output = pipe(messages, max_new_tokens=100, do_sample=False)
print(output[0]["generated_text"][-1]["content"])