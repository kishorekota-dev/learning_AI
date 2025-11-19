# Module 9: Tool Integration with MCP (The "USB-C" for AI)

## 1. The "Tower of Babel" Problem

Imagine if every time you bought a new mouse, you had to solder its wires directly onto your computer's motherboard. And if you bought a new computer, that mouse was useless because the wiring was different.

For a long time, this was the state of AI integration.
*   If you wanted **ChatGPT** to access your Google Drive, OpenAI had to build a specific "Google Drive Plugin".
*   If you wanted **Claude** to access your Google Drive, Anthropic had to build a *different* "Google Drive Tool".
*   If you wanted your custom **VS Code Extension** to access Google Drive, you had to write *yet another* integration.

This is the **"m × n" problem**: $m$ AI models $\times$ $n$ data sources = a nightmare of custom code.

## 2. Enter MCP: The Universal Standard

The **Model Context Protocol (MCP)**, introduced by Anthropic in late 2024, is an open standard that solves this. Think of it as **"USB-C for AI"**.

*   **Before USB**: You had a PS/2 port for keyboards, a parallel port for printers, and a serial port for mice.
*   **With USB**: You build a device with a USB connector, and it works with *any* computer.

**With MCP**: You build an **MCP Server** for your data (e.g., a "PostgreSQL Server"), and it instantly works with *any* MCP-compliant client (Claude Desktop, Cursor, Zed, or your own agent).

## 3. Architecture: Host, Client, and Server

MCP decouples the "Brain" (the AI) from the "Hands" (the Tools).

```mermaid
graph LR
    subgraph "The Host (User's Environment)"
        Host[MCP Host (e.g., Claude Desktop, IDE)]
        Client[MCP Client (Internal Connector)]
    end
    
    subgraph "The Servers (Data Sources)"
        S1[MCP Server: GitHub]
        S2[MCP Server: PostgreSQL]
        S3[MCP Server: Local Files]
    end
    
    Host -- "User asks: 'Check my PRs'" --> Client
    Client -- "JSON-RPC Protocol" --> S1
    Client -- "JSON-RPC Protocol" --> S2
    Client -- "JSON-RPC Protocol" --> S3
```

### The Three Primitives
An MCP Server exposes three things to the AI:
1.  **Resources**: Passive data that can be read. (e.g., "The contents of `logs.txt`").
2.  **Tools**: Functions that can be executed. (e.g., "Run SQL query", "Create GitHub Issue").
3.  **Prompts**: Pre-written templates to help the AI use the server. (e.g., "Analyze this database schema").

## 4. Building Your First MCP Server

Let's build a simple MCP server that gives an AI "Math Superpowers". We'll use the Python SDK.

### Prerequisites
```bash
pip install mcp
```

### The Code (`math_server.py`)

```python
from mcp.server.fastmcp import FastMCP

# 1. Initialize the Server
# We call it "Math Tools" - this is what the AI sees.
mcp = FastMCP("Math Tools")

# 2. Define a Tool
# The type hints (int, str) are CRITICAL. 
# MCP uses them to generate the JSON schema for the LLM.
@mcp.tool()
def calculate_compound_interest(principal: float, rate: float, years: int) -> str:
    """
    Calculates compound interest.
    Use this when the user asks about investment growth.
    """
    amount = principal * (1 + rate/100) ** years
    return f"After {years} years, ${principal} at {rate}% becomes ${amount:.2f}"

# 3. Define a Dynamic Resource
# This allows the AI to "read" the current configuration as if it were a file.
@mcp.resource("config://interest-rates")
def get_current_rates() -> str:
    """Returns the current market interest rates."""
    return "Savings: 4.5%, Stocks: 7.0%, Crypto: Volatile"

if __name__ == "__main__":
    # This runs the server over Standard IO (stdin/stdout)
    # This is the most secure way to run locally (no open network ports).
    mcp.run()
```

### How to Connect (The "USB" Moment)
You don't need to write a client code to test this. You can plug this directly into **Claude Desktop**:

1.  Open Claude Desktop config (`claude_desktop_config.json`).
2.  Add your server:
    ```json
    "mcpServers": {
      "my-math-server": {
        "command": "python",
        "args": ["/absolute/path/to/math_server.py"]
      }
    }
    ```
3.  Restart Claude. You can now ask: *"If I invest $1000 at current market savings rates for 10 years, what do I get?"*
    *   Claude will **read** the `config://interest-rates` resource to find the 4.5% rate.
    *   Claude will **call** the `calculate_compound_interest` tool.
    *   Claude will **answer** you.

## 5. The Ecosystem: Pre-built Servers

You don't always need to build. The open-source community is building "drivers" for everything:

*   **Filesystem Server**: Give AI safe access to read/write a specific folder.
*   **PostgreSQL Server**: Let AI query your database (with read-only permissions!).
*   **GitHub Server**: Let AI search your repos, read issues, and review PRs.
*   **Brave Search Server**: Give your local LLM access to the live internet via Brave's API.
*   **Memory Server**: A graph database that lets the AI "remember" facts about you across conversations.

## 6. Security: The "Human in the Loop"

The power of MCP is local execution, but that brings risks. If an AI has access to your file system, can it delete everything?

MCP solves this with **Host-Controlled Permissions**:
1.  **Local by Default**: MCP servers usually run on *your* machine, not a remote cloud. Your database password never leaves your laptop.
2.  **User Approval**: When the AI wants to execute a tool (like `delete_file` or `drop_table`), the Host (e.g., Claude Desktop) intercepts the request and asks you: *"The AI wants to run 'delete_file'. Allow?"*

## 7. Summary

MCP is the missing link that turns "Chatbots" into "System Integrators". By standardizing how AI connects to data, we move from building fragile pipelines to plugging in robust, reusable modules.

**Next Module**: Now that our AI can use tools, how do we make it *smart* enough to use them in complex sequences? We'll look at **Advanced Architectures & Reasoning Models**.

## References & Further Reading
*   **Anthropic Announcement**: [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
*   **Specification**: [MCP GitHub Repository](https://github.com/model-context-protocol)
*   **Community Servers**: [MCP Servers List](https://github.com/model-context-protocol/servers)
