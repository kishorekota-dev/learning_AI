# learning_AI

This repository contains a complete, narrative-style **Generative AI Learning Path** in Markdown, plus a few small runnable examples.

The main content lives in `genai_learning/` and is organized as numbered modules that you can read in order or dip into as needed.

## How to Use This Course

1. **Read in Order (Recommended)**  
	Start with `genai_learning/index.md`, which links all modules from **01** (Introduction) through **14** (Capstone Project).

2. **Run the Small Examples**  
	Inside `genai_learning/examples/` you’ll find tiny, focused scripts that mirror key ideas from the notes.
	- `minimal_rag.py`: smallest possible Retrieval-Augmented Generation example using OpenAI + Chroma (from Module 4).

3. **Treat it Like a Textbook + Lab**  
	- Use the Markdown modules as your “textbook”.
	- Use the `examples/` folder and your own experiments as the “lab”.

4. **Suggested Audience & Path**  
	- Audience: Engineers / technical PMs comfortable with basic Python and APIs.
	- Path: Modules **01–05** for foundations, **06–09** for multimodal + agents, **10–13** for advanced/enterprise, **14** as the capstone build.

## Quickstart: Minimal RAG Example

From the repo root:

```bash
cd genai_learning
pip install openai langchain-community langchain-openai chromadb
export OPENAI_API_KEY=your_key_here
python examples/minimal_rag.py
```

You should see an answer that combines facts from the tiny in-memory “corpus”, demonstrating the core RAG loop used throughout the material.
