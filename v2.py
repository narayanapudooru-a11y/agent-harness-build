import sys

sys.stdout.reconfigure(encoding='utf-8') # so emoji don't crash  the Windows console

import json
from pathlib import Path
from openai import OpenAI

# One Client. Point it at any OpenAI-compatible server (ollama here)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  
MODEL = "qwen3.5:2b"  # or "ollama:llama2-7b" or "ollama:llama2-13b"

WORKSPACE = Path(__file__).parent / "workspace"  # the folder where the user can read files

# --- the actual tools: plain Python functions ---------
def list_files() -> str:
    """
    List the files in the user's workspace folder.
    """
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"

def read_file(filename: str) -> str:
    """
    Read one file from the user's workspace folder.
    """
    # with open(filename, "r") as f:
    #     return f.read()
    path = WORKSPACE / filename
    if not path.is_file():
        return f"Error: {filename} is not a file."
    return path.read_text(encoding="utf-8")  # read the file as UTF-8 text

def write_file(filename: str, content: str) -> str:
    """
    Write content to a file in the user's workspace folder.
    """
    path = WORKSPACE / filename
    path.write_text(content, encoding="utf-8")  # write the file as UTF-8 text
    return f"Successfully wrote to {filename}."

TOOLS = {"list_files": list_files, "read_file": read_file, "write_file": write_file}

# --- what the models sees: a JSON description of each tool ----------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read."
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file."
                    }
                },
                "required": ["filename", "content"]
            }
        }
    }
]

def run_agent(user_message: str) -> str:
    """
    Send a message to the model and return its response.
    """
    messages = [
            {"role": "system", "content": "You are a helpful personal assistant."},
            {"role": "user", "content": user_message}
        ]

    # The agent loop. this is the whole trick
    while True:

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS
        )
        message = response.choices[0].message

        # The model didn't want a tool, so just answer the user directly
        if not message.tool_calls:
            return message.content or "" # done - it answered

        # Run the tool it asked for and give it the result... Once
        messages.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f" [tool] {call.function.name}({args})")
            result = TOOLS[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

if __name__ == "__main__":
    print(f"v0 assistant ({MODEL}) - ctrl+c to quit")
    try:
        while True:
            user_input = input("\nYou: ")  
            print("\nassistant: ", run_agent(user_input))
    except (EOFError,KeyboardInterrupt):
        print("\nGoodbye!")