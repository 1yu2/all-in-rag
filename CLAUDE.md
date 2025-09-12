# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Dependencies
```bash
# Install main dependencies
pip install -r code/requirements.txt

# Start required services (Milvus vector database)
cd code && docker-compose up -d

# Chapter-specific installations
pip install -r code/C8/requirements.txt  # Basic RAG
pip install -r code/C9/requirements.txt  # Advanced Graph RAG
```

### Running Applications
```bash
# Advanced Graph RAG system (Chapter 9)
cd code/C9 && python main.py

# Basic RAG system (Chapter 8)
cd code/C8 && python main.py

# Run individual chapter examples
cd code/C[chapter-number] && python main.py
```

## Architecture Overview

### Educational Tutorial Structure
- **10 Chapters** from RAG basics to advanced implementations
- **Bilingual documentation**: `/docs/` (Chinese/English)
- **Chapter-based code**: `/code/C1/` through `/code/C10/`

### Key Components
- **Chapter 9**: Advanced Graph RAG with modular architecture
- **Chapter 8**: Basic RAG implementation
- **Chapter 3**: Vector embeddings and multimodal processing
- **Data**: `/data/` sample datasets
- **Models**: `/models/` pre-trained model files

### Technology Stack
- **Python 3.12.7** with PyTorch 2.6.0 ecosystem
- **LangChain 0.3.26+** and **LlamaIndex 0.12.51+** RAG frameworks
- **Vector Databases**: Milvus (primary), FAISS-CPU, ChromaDB
- **Graph Database**: Neo4j 5.0.0+ for Graph RAG
- **Document Processing**: Unstructured, PyPDF, multimodal support

## Development Guidelines

### Graph RAG System (Chapter 9)
The most complex implementation uses:
- **Modular architecture** with 7 specialized modules
- **Intelligent query routing** for optimal retrieval strategy selection
- **Hybrid retrieval** combining vector search with graph-based retrieval
- **Interactive CLI** with stats, rebuild, and testing capabilities
- **Configuration management** via `config.py`

### Code Organization Patterns
- Follow existing chapter structure for new implementations
- Use modular design patterns from Chapter 9 as reference
- Maintain bilingual documentation consistency
- Implement proper error handling and logging patterns

### Configuration
- Main config: `/code/C9/config.py`
- Environment variables for API keys (OpenAI, DeepSeek)
- Docker-based infrastructure setup in `/code/docker-compose.yml`