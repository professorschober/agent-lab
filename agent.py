import ast
import json
import math
import operator
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = """You are a code analysis agent. Given a goal, use the available tools to inspect files and return concise, factual answers. Always cite the file paths you read. Never invent file contents."""


class Memory:
    """Separate connections; completed histories are saved atomically."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS sessions (project TEXT, session TEXT, history TEXT NOT NULL, updated TEXT, PRIMARY KEY(project, session))")
            db.execute("CREATE TABLE IF NOT EXISTS facts (project TEXT, key TEXT, value TEXT NOT NULL, source TEXT NOT NULL, updated TEXT, PRIMARY KEY(project, key))")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()


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

def calculate(expression: str) -> str:
    """Bounded arithmetic AST; never executes Python code."""
    if not expression or len(expression) > 200:
        raise ValueError("Ausdruck muss 1–200 Zeichen enthalten")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ValueError("Ausdruck ist zu komplex")
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent muss zwischen -12 und 12 liegen")
            result = operations[type(node.op)](left, right)
        else:
            raise ValueError("Nur Zahlen, Klammern und + - * / // % ** erlaubt")
        if type(result) not in (int, float) or not math.isfinite(result) or abs(result) > 1e100:
            raise ValueError("Ergebnis außerhalb des erlaubten Zahlenbereichs")
        return result

    return json.dumps({"expression": expression, "result": visit(tree.body)})


# Tool registry: maps tool names to functions
TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "calculate": calculate,
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
    },
    {
        "name": "calculate",
        "description": "Evaluate bounded arithmetic without executing Python code. Returns JSON with the expression and result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression of 1–200 characters using numbers, parentheses and + - * / // % **. Exponents must be between -12 and 12."
                }
            },
            "required": ["expression"]
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
                instructions=SYSTEM_PROMPT,
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
                print(f"Tool call: {tool_name} | Parameters: {call.arguments}", flush=True)
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
