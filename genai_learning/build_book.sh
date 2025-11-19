#!/usr/bin/env bash
set -e

cat 01_Introduction_to_GenAI.md \
    02_NLP_Fundamentals.md \
    03_Advanced_Prompt_Engineering.md \
    04_RAG_and_Vector_Stores.md \
    05_Model_Customization_and_Evaluation.md \
    06_Transcriptions_and_Audio.md \
    07_Multimodal_Generation.md \
    08_Agentic_AI_with_LangGraph.md \
    09_Tool_Integration_with_MCP.md \
    10_Advanced_Architectures.md \
    11_SLM_and_LLMOps.md \
    12_Security_and_Enterprise_Strategy.md \
    13_Cloud_Implementation_Guide.md \
    14_Capstone_Project.md \
    > genai_learning_book.md
