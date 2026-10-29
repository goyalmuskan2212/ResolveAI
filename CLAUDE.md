# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Support Agent that processes customer support tickets — classifies them, retrieves knowledge base context via RAG, generates responses, and escalates when necessary. Built with LangGraph (orchestration), LlamaIndex (RAG), Qdrant (vector DB), Langfuse (tracing), FastAPI (API), Streamlit (UI), and Google Gemini (LLM).

## Virtual Environment

This project uses a Python virtual environment (`venv/`). **All commands must be run from the activated venv.** Before running any `python`, `pip`, `pytest`, `uvicorn`, or `streamlit` command, activate the environment:

```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

Never install packages globally — always ensure the venv is active. If a package appears missing, check that the venv is activated before installing anything.

## Commands

```bash
# Install dependencies (venv must be active)
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run a single test
pytest tests/test_classification.py -v -k "test_name"

# Start local infrastructure (Qdrant, Langfuse, PostgreSQL)
docker-compose --profile local up -d

# Start API server (when implemented)
uvicorn app.api:app --reload

# Start Streamlit UI (when implemented)
streamlit run app/ui/app.py
```

## Architecture

### LangGraph State Machine

The agent is a state graph with 8 nodes and conditional routing:

```
classify → route_by_category → {
    "question"  → rag_search → generate → quality_check → route_by_quality
    "complaint" → empathy → rag_search → generate → quality_check → route_by_quality
    "unclear"   → clarify → END (with interrupt for human input)
}

route_by_quality → {
    "pass"  → send_reply → END
    "retry" → rag_search (max 2 iterations)
    "fail"  → escalate → END
}
```

### Key Architectural Decisions

- **AR-1 (Empathy Separation):** Empathy node only sets `tone_instructions` and `empathy_preamble` in state — it does NOT do RAG. All paths share the same rag_search → generate → quality_check pipeline.
- **AR-2 (Model Split):** Gemini 2.5 Flash for classify/quality_check (cheaper); Gemini 3.0 Flash for generate/empathy (better quality).
- **AR-3 (HTTP Boundary):** Streamlit communicates with FastAPI via REST, not direct imports.
- **AR-4 (Early Escalation):** Three-layer RAG quality gate: (1) SimilarityPostprocessor filters low-cosine chunks, (2) LLMRerank scores chunks 1-10 — if best score < 3.0, escalate, (3) LLM relevance verification checks if context contains a direct answer. If any layer fails → escalate immediately, skip generate + quality_check to avoid hallucinations.

### Node Function Convention

Every graph node follows this signature:

```python
def node_name(state: AgentState) -> dict:
    return {"field1": value1, "field2": value2}
```

### State Definition

`app/agent/state.py` defines `AgentState` as a TypedDict with 14 fields covering: dialog/input, classification results, RAG results, generation/quality, and routing/control flow. Uses LangGraph's `Annotated[list, add_messages]` for message history.

### Configuration

`app/config.py` uses Pydantic `BaseSettings` loading from `.env`. Singleton: `from app.config import settings`. All API keys are `repr=False`.

### RAG Setup

`app/rag/qdrant_store.py` provides factory functions `get_qdrant_client()` and `get_vector_store()`. LlamaIndex uses SentenceSplitter (chunk_size=512, overlap=50). Query engine: top_k=10, SimilarityPostprocessor (cutoff=0.5), LLMRerank (top_n=5), response_mode="no_text" (no synthesis — only source nodes are used).

## Project Spec & Roadmap

- `project_2_support_agent.md` — full specification (691 lines)
- `roadmap.md` — block-by-block implementation plan
