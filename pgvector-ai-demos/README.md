---
title: AI Search with pgvector
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: apache-2.0
---

# AI Search Application with pgvector

An advanced AI-powered search application using Aurora PostgreSQL with pgvector 0.8.0, featuring:

- **AI Search**: Vector similarity search with re-ranking and query rewriting
- **Multimodal RAG**: Explore multimodal RAG over complex documents 
- **Agentic RAG**: AI agent enhanced search experience
- **ColBERT**: Multi-vector search implementation

## Features

- Semantic search with vector embeddings stored in Aurora PostgreSQL
- Multimodal search (text + images)
- ColBERT multi-vector search
- Hybrid search capabilities
- Query rewriting with LLM
- Re-ranking with cross-encoders
- Document analysis with tables and images
- User behavior insights tracking

## Technology Stack

- **Database**: Aurora PostgreSQL with pgvector 0.8.0
- **Embeddings**: Amazon Bedrock (Titan models)
- **LLM**: Amazon Bedrock (Claude models)
- **Vector Search**: pgvector with HNSW and IVF indexes
- **Frontend**: Streamlit
- **Multi-vector**: ColBERT implementation

## Architecture

- Vector similarity search using pgvector
- Document processing and chunking
- Multi-modal embeddings (text + images)
- Real-time search with caching
- User behavior analytics