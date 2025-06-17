# Policy QA RAG Application

A Retrieval-Augmented Generation (RAG) system for querying organizational policy documents using natural language.

## Document Corpus
The application processes four policy documents:
- IT Policy (3 pages): Covers asset management, damage costs, security protocols
- Rewards and Recognition Policy (3 pages): Details awards systems and FAQs
- Leave Policy (3 pages): Documents all leave types and entitlements  
- Staff Loan Policy (3 pages): Outlines loan procedures and eligibility

## System Architecture

### Core Components
- Document Loader: PDF parsing with metadata extraction
- Text Chunking: Recursive semantic segmentation
- Embedding Model: Sentence Transformers for dense vector representations
- Vector Store: FAISS for efficient similarity search
- LLM Interface: GPT-4.o via OpenAI API

## Setup Instructions

### Environment Configuration
1. Create Conda environment:
   conda create --name rag_env python=3.10
   conda activate rag_env
2. Install dependencies:
   pip install -r requirements.txt

### Operational Modes
#### Index Building Mode

Initializes document processing pipeline:
- PDF text extraction with metadata preservation
- Recursive text chunking with configurable overlap
- Vector embedding generation
- FAISS index creation
    
    python main1.py --mode build

#### Query Mode

Interactive console for policy questions:
- Natural language query processing
- MMR-based document retrieval
- Context-aware answer generation
- Source attribution display
    python main1.py --mode query

## Technical Implementation

### Chunking Strategy
- Recursive text splitting algorithm
- Configurable chunk size (1000 chars)
- Context-preserving overlap (150 chars)

### Retrieval Pipeline
- We have used Maximal Marginal Relevance (MMR) retrieval for accurate Semantic Search
- Top-k document fetching with fetch-k expansion
- Metadata-aware result ranking

### QA Chain Configuration
- "Stuff" document chain type
- Custom prompt templating
- Source document verification

## Key Features

**Metadata-preserving document ingestion**

**Context-aware text segmentation**

**Local vector store implementation**

**Diversity-optimized retrieval**

**Verifiable answer generation**

**Dual-mode operation (build/query)**
