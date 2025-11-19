"""Streamlit UI for the minimal RAG demo.

Mirrors Module 4 (RAG & Vector Stores):
- Tiny in-memory corpus
- Chroma vector store
- OpenAI LLM via LangChain

Usage:
    pip install streamlit openai langchain-community langchain-openai chromadb
    export OPENAI_API_KEY=sk-...
    streamlit run rag_app_streamlit.py
"""

import os

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


@st.cache_resource
def build_rag_chain() -> RetrievalQA:
    texts = [
        "Our product GenX launches in Q4 2024.",
        "The company was founded in 2010.",
        "We use Retrieval Augmented Generation (RAG) for internal search.",
    ]

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(texts, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def main() -> None:
    st.title("Minimal RAG Demo")
    st.write("Ask questions about a tiny in-memory knowledge base.")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Please set OPENAI_API_KEY in your environment before running.")
        st.stop()

    qa = build_rag_chain()

    query = st.text_input("Your question", "When does GenX launch and when was the company founded?")

    if st.button("Ask") and query.strip():
        with st.spinner("Thinking..."):
            result = qa.invoke(query)
        st.subheader("Answer")
        st.write(result["result"])


if __name__ == "__main__":
    main()
