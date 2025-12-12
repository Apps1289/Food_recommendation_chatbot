from huggingface_hub import InferenceClient

# Paste your token here
MY_TOKEN = "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"

# Initialize the client
client = InferenceClient(api_key=MY_TOKEN)

# Send a simple message
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[{"role": "user", "content": "Hello! Can you help me with a diet?"}],
    max_tokens=100
)

print(response.choices[0].message.content)