# OpenAI API Setup Guide

The direct OpenAI API is an alternative to Azure OpenAI for LLM operations. This project uses it for:
- **Generation** — Producing RAG answers from retrieved context
- **Judging** — Evaluating answer quality (relevance, faithfulness)

> **Note:** This project uses Azure OpenAI by default when `AZURE_OPENAI_ENDPOINT` is set. To use direct OpenAI API instead, ensure `AZURE_OPENAI_ENDPOINT` is **not** set in your environment.

## Prerequisites

- An OpenAI account ([Sign up](https://platform.openai.com/signup))

---

## Setup Steps

### 1. Create an OpenAI Account

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in

### 2. Add Payment Method

1. Go to **Settings** → **Billing**
2. Add a payment method
3. (Optional) Set usage limits to control costs

### 3. Generate an API Key

1. Go to **API Keys** ([direct link](https://platform.openai.com/api-keys))
2. Click **Create new secret key**
3. Give it a name (e.g., `rag-eval`)
4. Copy the key immediately — it won't be shown again

---

## Environment Variables Reference

| Variable         | Required | Default | Description         |
| ---------------- | -------- | ------- | ------------------- |
| `OPENAI_API_KEY` | ✅*       | —       | Your OpenAI API key |

*Required only if `AZURE_OPENAI_ENDPOINT` is not set.

### Example Configuration

```bash
# .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Make sure Azure OpenAI is NOT configured (comment out or remove):
# AZURE_OPENAI_ENDPOINT=...
```

---

## LLM Provider Selection Logic

This project automatically selects between Azure OpenAI and direct OpenAI:

```
If AZURE_OPENAI_ENDPOINT is set → Use Azure OpenAI
Otherwise → Use direct OpenAI API
```

To use direct OpenAI:
1. Set `OPENAI_API_KEY` in your `.env`
2. Ensure `AZURE_OPENAI_ENDPOINT` is **not** set (remove or comment it out)

---

## Model Configuration

The default model used in the notebooks is `gpt-4o`. To use a different model, modify the LLM initialization in the notebook cells:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")  # or another model
```

### Supported Models

| Model         | Use Case                              | Cost   |
| ------------- | ------------------------------------- | ------ |
| `gpt-4o`      | Best quality, recommended for judging | Higher |
| `gpt-4o-mini` | Good quality, lower cost              | Lower  |
| `gpt-4-turbo` | High quality, large context           | Higher |

---

## Verification

Test your OpenAI API connection:

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

You should receive a JSON response with a completion.

---

## Cost Management

Evaluation runs can incur significant LLM costs due to judging calls. To manage costs:

1. **Start small** — Run evaluations on a small dataset first
2. **Use cheaper models** — `gpt-4o-mini` is significantly cheaper for judging
3. **Set usage limits** — Configure spending caps in the OpenAI dashboard
4. **Monitor usage** — Check the [Usage page](https://platform.openai.com/usage) regularly

---

## Official Documentation

- [OpenAI Platform](https://platform.openai.com/)
- [API Reference](https://platform.openai.com/docs/api-reference)
- [Models Overview](https://platform.openai.com/docs/models)
- [Pricing](https://openai.com/pricing)
- [Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
