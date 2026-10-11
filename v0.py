import sys

sys.stdout.reconfigure(encoding='utf-8') # so emoji don't crash  the Windows console

from openai import OpenAI

# One Client. Point it at any OpenAI-compatible server (ollama here)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  
MODEL = "qwen3.5:2b"  # or "ollama:llama2-7b" or "ollama:llama2-13b"

def chat(user_message: str) -> str:
    """
    Send a message to the model and return its response.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful personal assistant."},
            {"role": "user", "content": user_message}
        ],
        # max_tokens=200,
    )
    return response.choices[0].message.content or ""

if __name__ == "__main__":
    print(f"v0 assistant ({MODEL}) - ctrl+c to quit")
    try:
        while True:
            user_input = input("\nYou: ")  
            print("\nassistant: ", chat(user_input))
    except (EOFError,KeyboardInterrupt):
        print("\nGoodbye!")