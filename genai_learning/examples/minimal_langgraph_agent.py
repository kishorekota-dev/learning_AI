"""Minimal LangGraph-style agent skeleton.

This mirrors concepts from Module 8 (Agentic AI with LangGraph):
- A simple state with `messages`.
- A single tool node (a fake calculator) and a router.

This is intentionally lightweight and meant for reading alongside the module,
not as a production implementation.
"""

from typing import TypedDict, List


class Message(TypedDict):
    role: str
    content: str


class State(TypedDict):
    messages: List[Message]


def tool_calculator(query: str) -> str:
    """Very fake calculator: recognizes only one operation.

    In a real LangGraph app, this would be a proper tool function
    (e.g., SQL query, web search, etc.).
    """

    if "2+2" in query.replace(" ", ""):
        return "4"
    return "I can only calculate 2+2 in this demo."


def supervisor(state: State) -> State:
    """Very small supervisor.

    - Looks at the last user message.
    - Decides whether to call the `tool_calculator`.
    - Appends the answer as an assistant message.
    """

    messages = state["messages"]
    last = messages[-1]

    if last["role"] == "user" and "2+2" in last["content"]:
        result = tool_calculator(last["content"])
        messages.append({"role": "assistant", "content": f"Tool result: {result}"})
    else:
        messages.append({"role": "assistant", "content": "I am a tiny demo supervisor. Try asking 'What is 2+2?'"})

    return {"messages": messages}


def main() -> None:
    state: State = {"messages": []}

    user_query = "What is 2 + 2?"
    state["messages"].append({"role": "user", "content": user_query})

    new_state = supervisor(state)

    print("Conversation:")
    for m in new_state["messages"]:
        print(f"{m['role']}: {m['content']}")


if __name__ == "__main__":
    main()
