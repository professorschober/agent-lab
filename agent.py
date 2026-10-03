import json
import os

from openai import OpenAI

def list_files(directory: str) -> str:
    """List files in a directory."""
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except Exception as e:
        return f"Error: {e}"

def read_file(path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

# Tool registry: maps tool names to functions
TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
}

TOOL_DEFINITIONS = [
    {
        "name": "list_files",
        "description": "List the files in a directory. Returns a newline-separated list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Absolute or relative path to the directory."
                }
            },
            "required": ["directory"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file contents as a string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file."
                }
            },
            "required": ["path"]
        }
    }
]


def run_agent(goal: str, max_steps: int = 10) -> str:
    """Run an agent loop until the goal is reached or max_steps exceeded."""
    tools = [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                **tool["input_schema"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        for tool in TOOL_DEFINITIONS
    ]
    messages = [{"role": "user", "content": goal}]

    with OpenAI() as client:
        for step in range(max_steps):
            response = client.responses.create(
                model="gpt-5-mini",
                max_output_tokens=4096,
                tools=tools,
                input=messages,
            )

            # Preserve the full output, including reasoning and tool calls.
            messages.extend(response.output)
            if response.status != "completed":
                return f"Agent response did not complete (status: {response.status})."

            tool_calls = [
                item for item in response.output if item.type == "function_call"
            ]
            if not tool_calls:
                return response.output_text or "Agent finished with no text output."

            for call in tool_calls:
                tool_name = call.name
                if tool_name in TOOLS:
                    try:
                        tool_input = json.loads(call.arguments)
                        result = TOOLS[tool_name](**tool_input)
                    except Exception as e:
                        result = f"Tool '{tool_name}' raised an error: {e}. Try a different approach."
                else:
                    result = f"Error: tool '{tool_name}' not found. Available tools: {list(TOOLS.keys())}"

                messages.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": str(result),
                })

    return "Max steps reached without completion."


if __name__ == "__main__":
    result = run_agent("List the Python files in the current directory and summarize what the largest one does.")
    print(result)
