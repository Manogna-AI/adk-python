# TC-AI Nexus

A role-oriented **Human + AI** multi-agent framework for Teamcenter engineering excellence, built with:

- **Google Agent Development Kit (ADK)** for multi-agent orchestration.
- **Gradio** for the front-end chat console.
- **GPT-5.3 via LiteLLM** (`openai/gpt-5.3` by default).
- **RAG** over sample Siemens/Teamcenter knowledge.
- **MCP-ready tools** for secure live enterprise context integration.
- **Open-source deployment stack**: Docker, Docker Compose, Kubernetes manifests.

---

## 1) Architecture

```text
Human Engineer
   |
   v
Gradio UI ---> Orchestrator Agent (ADK)
                 |
                 +--> Planning Agent
                 +--> Server-Side Agent (ITK/SOA)
                 +--> BMIDE Agent
                 +--> UI Agent
                 +--> DevOps Agent
                 +--> Production Support Agent

All agents can call:
- RAG tool (`collect_context`) over indexed Teamcenter docs
- Optional MCP toolsets for live code/log/db access
```

### Human-in-the-loop policy

AI proposes analysis + steps, while humans approve and execute production actions.

---

## 2) Files

- `agent.py`: Orchestrator, planning, and role agents.
- `rag_index.py`: Lightweight local embedding + vector search.
- `data/seed_siemens_docs.json`: Sample Siemens Teamcenter documents.
- `scripts/build_sample_embeddings.py`: Generates sample embedding vectors.
- `gradio_app.py`: Gradio front-end.
- `main.py`: CLI runner.
- `deploy/`: Docker + compose + Kubernetes templates.

---

## 3) Quick start

```bash
cd contributing/samples/tc_ai_nexus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="<your-key>"
export TC_AI_MODEL="openai/gpt-5.3"
python scripts/build_sample_embeddings.py
python gradio_app.py
```

Open: `http://localhost:7860`

---

## 4) MCP integration (optional)

Set MCP stdio servers via `TC_AI_MCP_STDIO_JSON`:

```bash
export TC_AI_MCP_STDIO_JSON='[
  {
    "command": "python",
    "args": ["/path/to/enterprise_mcp_server.py"],
    "allow": ["search_logs", "read_repo_file", "run_safe_query"]
  }
]'
```

The orchestrator and role agents automatically receive those tools.

---

## 5) Deployment

### Docker Compose

```bash
cd deploy
docker compose up --build
```

### Kubernetes

```bash
kubectl apply -f deploy/k8s.yaml
```

---

## 6) LangChain / LangGraph extension points

This sample keeps core orchestration in ADK. You can additionally:

- Use LangChain loaders/splitters for production RAG ingestion.
- Use LangGraph for deterministic pre-check workflows before delegating to ADK agents.
- Persist vectors in Qdrant (included in compose) for larger corpora.
