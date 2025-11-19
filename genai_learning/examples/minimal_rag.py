"""Minimal RAG example using OpenAI + Chroma.

This is intentionally tiny and mirrors concepts from:
- Module 4 (RAG & Vector Stores)

Prerequisites:
    pip install openai langchain-community langchain-openai chromadb

Set env vars before running:
    export OPENAI_API_KEY=sk-...

Run:
    python minimal_rag.py
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


def build_rag_chain():
    # 1) Toy "corpus" – in real life, this is your documents
    texts = [
        "Our product GenX launches in Q4 2024.",
        "The company was founded in 2010.",
        "We use Retrieval Augmented Generation for internal search.",
    ]

    # 2) Embed + store in a local Chroma DB (in-memory by default)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(texts, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # 3) LLM + RetrievalQA chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY before running.")

    qa = build_rag_chain()

    query = "When does GenX launch, and when was the company founded?"
    result = qa.invoke(query)

    print("Question:")
    print(query)
    print("\nAnswer:")
    print(result["result"])


if __name__ == "__main__":
    main()
