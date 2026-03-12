#!/usr/bin/env bash
# ==============================================================================
# Setup script for Customer Support Agent — Evaluation Framework
# ==============================================================================
set -euo pipefail

echo "==========================================="
echo " Customer Support Agent — Setup"
echo "==========================================="
echo ""

# ── 1. Check / install UV ────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    echo "📦 UV not found — installing…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
echo "✅ UV detected ($UV_VERSION)"

# ── 2. Ensure Python 3.11+ (UV-managed) ──────────────────────────────────
uv python install 3.11 >/dev/null 2>&1
echo "✅ Python 3.11 available (UV-managed)"

# ── 3. Create .env from template ─────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📄 Created .env from template"
    echo "   ⚠️  Edit .env with your credentials before running the app"
else
    echo "📄 .env already exists — skipping"
fi

# ── 4. Install dependencies (UV auto-selects Python 3.11) ────────────────
echo ""
echo "📦 Installing dependencies with UV…"
uv sync

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "==========================================="
echo " ✅ Setup Complete"
echo "==========================================="
echo ""
echo " Next steps:"
echo ""
echo "   1. Edit .env with your credentials:"
echo "      - NESTLE_CLIENT_ID"
echo "      - NESTLE_CLIENT_SECRET"
echo "      - DATABRICKS_HOST"
echo "      - ARM_CLIENT_ID"
echo "      - ARM_CLIENT_SECRET"
echo "      - ARM_TENANT_ID"
echo "      - WAREHOUSE_ID"
echo "      - DATABRICKS_CATALOG  (optional, default: samples)"
echo "      - DATABRICKS_SCHEMA   (optional, default: bakehouse)"
echo "      - LANGFUSE_PUBLIC_KEY  (optional)"
echo "      - LANGFUSE_SECRET_KEY  (optional)"
echo ""
echo "   2. Start the MCP server:"
echo "      uv run python -m agent_eval.mcp_server.server"
echo ""
echo "   3. Launch the chat UI (separate terminal):"
echo "      uv run streamlit run agent_eval/ui/app.py"
echo ""
echo "   4. Run evaluations (no MCP server needed):"
echo "      uv run python -m evaluation.runner"
echo ""
