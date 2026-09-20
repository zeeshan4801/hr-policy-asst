# 📘 HR Policy Assistant - RAG Application


A Retrieval Augmented Generation (RAG) based HR assistant built with:

- Streamlit
- FAISS Vector Database
- Sentence Transformers
- PyMuPDF
- Groq LLM (openai/gpt-oss-20b)


## Features

✅ Upload HR Policy PDF

✅ Extract policy information

✅ Create semantic search database

✅ Ask HR related questions

✅ Answers grounded in uploaded documents

✅ No hallucination approach



## How it Works


PDF Upload

↓

Text Extraction

↓

Text Chunking

↓

Embedding Generation

↓

FAISS Similarity Search

↓

Groq LLM Response



## Deployment

This application can be deployed using:

GitHub + Streamlit Cloud



## Required API

You need:

Groq API Key

Get it from:

https://console.groq.com


## Local Run

Install requirements:

pip install -r requirements.txt


Run:

streamlit run app.py



## Technologies

Python

RAG

LLM

Vector Database

Natural Language Processing
