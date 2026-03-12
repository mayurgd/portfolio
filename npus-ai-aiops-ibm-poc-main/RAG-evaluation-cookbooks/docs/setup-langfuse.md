# Langfuse Setup Guide

Langfuse is an open-source LLM observability platform used by this project for:
- **Tracing** — Capturing RAG pipeline execution spans
- **Datasets** — Storing evaluation Q&A pairs
- **Scoring** — Recording evaluation metrics against traces

## Option A: Langfuse Cloud (Recommended for Getting Started and using Public Datasets)

### 1. Create an Account

1. Go to [Langfuse Cloud](https://cloud.langfuse.com/)
2. Sign up with GitHub, Google, or email
3. Create a new organization (e.g., `RAG Eval`)
4. Create a new project in that organization (e.g., `rag-eval`)

### 2. Generate API Keys

1. Navigate to **Settings** → **API Keys**
2. Click **Create new API keys**
3. Provide a name (e.g., `rag-eval-keys`) and create
4. Copy both the **.env** content

### 3. Configure Environment Variables

Pase the copied content into your `.env` file:
```bash
# simple example
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

---

## Option B: Self-Hosted (Docker)

For production deployments or data residency requirements, you can self-host Langfuse.

### Prerequisites

- Docker and Docker Compose installed
- PostgreSQL database (or use the bundled one)

### 1. Clone and Start Langfuse

```bash
# Clone the Langfuse repository
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# Start with Docker Compose
docker compose up -d
```

Langfuse will be available at `http://localhost:3000`.

### 2. Create Project and API Keys

1. Open `http://localhost:3000` in your browser
2. Create an account and project
3. Generate API keys from **Settings** → **API Keys**

### 3. Configure Environment Variables

```bash
# .env
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE_BASE_URL=http://localhost:3000
```

### Alternative: Docker One-Liner

```bash
docker run -d \
  --name langfuse \
  -p 3000:3000 \
  -e DATABASE_URL="postgresql://postgres:postgres@host.docker.internal:5432/langfuse" \
  -e NEXTAUTH_SECRET="your-secret-key" \
  -e NEXTAUTH_URL="http://localhost:3000" \
  langfuse/langfuse:latest
```

---

## Official Documentation

- [Langfuse Documentation](https://langfuse.com/docs)
- [Self-Hosting Guide](https://langfuse.com/docs/deployment/self-host)
- [Python SDK Reference](https://langfuse.com/docs/sdk/python)
- [LangChain Integration](https://langfuse.com/docs/integrations/langchain)
