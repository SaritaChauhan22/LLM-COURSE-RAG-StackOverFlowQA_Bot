# StackOverflow RAG Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, Hugging Face, and Streamlit. It retrieves relevant programming questions from a Stack Overflow dataset and generates context-aware answers using a Large Language Model.

---

## Technologies

- Python
- LangChain
- Hugging Face
- FAISS
- Streamlit
- Sentence Transformers
- FLAN-T5

---

## Features

- Semantic Search
- Vector Database
- Related StackOverflow Questions
- Tags Display
- Source References
- Local Deployment

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

For public access, start the app with:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

This makes the app reachable from any browser on the deployed host.

---

## Project Workflow

CSV

↓

Documents

↓

Chunking

↓

Embeddings

↓

FAISS

↓

Retriever

↓

Prompt

↓

FLAN-T5

↓

Answer

---

## Dataset

StackOverflow QA Dataset

---

## Author

Your Name