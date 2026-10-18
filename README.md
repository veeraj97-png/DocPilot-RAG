# DocPilot — AI-Powered Technical Knowledge Assistant

DocPilot is a Python-based Retrieval-Augmented Generation (RAG)
application that answers questions using information retrieved
from technical documents.

The initial prototype uses the Triumph 400-series Owner's Handbook
as its knowledge base, with a focus on the Triumph Speed 400.

## Project Objective

Explore how Retrieval-Augmented Generation can make lengthy technical
documents easier to navigate while preserving source references.

The application is designed as an industry-independent RAG prototype
that can be adapted to other document-based knowledge domains.

## Technology Stack

- Python 3.12
- LangChain — Document chunking
- Nebius Qwen3-Embedding-8B — Text embeddings
- Pinecone — Vector database and semantic retrieval
- DeepSeek-V4-Flash-0731 via Nebius — Answer generation
- Streamlit — Web interface
- PyPDF — PDF text extraction

## RAG Architecture

PDF Document
    |
    v
Text Extraction (PyPDF)
    |
    v
Document Chunking (LangChain)
    |
    v
Embedding Generation (Nebius)
    |
    v
Vector Storage (Pinecone)
    |
    v
Semantic Retrieval
    |
    v
LLM Answer Generation (DeepSeek)
    |
    v
Answer with Source Page References

## Current Implementation

- PDF document ingestion
- Page-aware text chunking
- Embedding generation
- Pinecone vector storage and similarity search
- Retrieval of relevant document passages
- AI-generated answers grounded in retrieved text
- Original handbook page references
- Basic insufficient-information handling
- Browser-based interface using Streamlit

## Evaluation

The prototype was evaluated using 15 manually selected questions
covering technical specifications, maintenance, safety information,
model-specific applicability, and unsupported diagnostic requests.

Results:
- 12 primary-objective passes
- 2 partial passes
- 1 pass with retrieval and completeness limitations

These results are preliminary functional observations, not
a statistically validated accuracy measure.

## Key Findings

1. Semantic similarity does not always rank the most useful passage first.
2. PDF extraction can lose important table and heading structure.
3. Model-specific technical instructions require careful handling.
4. Retrieval can miss information needed for a complete answer.
5. Completion-token limits can prevent an LLM from producing a response.
6. Source-grounded answers still require verification.

## Known Limitations

- The application currently uses a fixed, pre-indexed knowledge base.
- PDF tables and diagrams are not fully reconstructed.
- Model-specific applicability is not enforced programmatically.
- Generated citations are not independently validated.
- Some answers may omit relevant information.
- The system is not a substitute for manufacturer-approved servicing
  or inspection by qualified technicians.

## Running Locally

1. Clone this repository.
2. Create a Python 3.12 virtual environment.
3. Install the packages from `requirements.txt`.
4. Create a `.env` file containing:
   - `NEBIUS_API_KEY`
   - `PINECONE_API_KEY`
5. Configure a Pinecone dense index with 4096 dimensions
   and cosine similarity.
6. Obtain an authorized copy of the source documentation
   and place it in the `data` directory.
7. Run `upload_pdf_vectors.py` to index the document.
8. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

The current application expects the index `docpilot-index`
and namespace `triumph-speed400`.

## Documentation and Copyright

The Triumph Owner's Handbook is not included in this repository.

The handbook belongs to its respective copyright owner.
Users should obtain documentation from authorized sources
and comply with applicable usage permissions.

## Future Improvements

- Improved PDF table and layout extraction
- Model-specific metadata filtering
- Retrieval reranking
- Automated citation validation
- Support for multiple documents
- More comprehensive evaluation

## Project Background

Built as a hands-on Python and RAG learning project during
the Mastering Agentic AI certification program.

The focus was on understanding document ingestion,
embeddings, vector databases, retrieval, generation,
and evaluation through practical implementation.