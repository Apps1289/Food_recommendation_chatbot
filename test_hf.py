import os
from huggingface_hub import InferenceClient

# Read token from environment (do NOT hardcode secrets)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN env var. Set it before running.")

# Initialize the client
client = InferenceClient(api_key=HF_TOKEN)

# Send a simple message
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[{"role": "user", "content": "Hello! Can you help me with a diet?"}],
    max_tokens=100
)

print(response.choices[0].message.content)
