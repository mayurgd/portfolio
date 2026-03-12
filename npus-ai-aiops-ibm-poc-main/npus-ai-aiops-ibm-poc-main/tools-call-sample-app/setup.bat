@echo off
REM ============================================================================
REM Setup script for Customer Support Agent — Evaluation Framework
REM ============================================================================

echo ===========================================
echo  Customer Support Agent — Setup
echo ===========================================
echo.

REM ── 1. Check / install UV ───────────────────────────────────────────────
set "UV_DIR=%USERPROFILE%\.local\bin"
set "PATH=%UV_DIR%;%PATH%"

uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [..] UV not found — installing...
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo [X] Failed to install UV. Install manually: https://docs.astral.sh/uv/
        exit /b 1
    )
)

for /f "tokens=*" %%v in ('uv --version 2^>^&1') do set UV_VER=%%v
echo [OK] UV detected ^(%UV_VER%^)

REM ── 2. Ensure Python 3.11+ (UV-managed) ────────────────────────────────
uv python install 3.11 >nul 2>&1
echo [OK] Python 3.11 available (UV-managed)

REM ── 3. Create .env from template ────────────────────────────────────────
if not exist .env (
    copy .env.example .env >nul
    echo [OK] Created .env from template
    echo     WARNING: Edit .env with your credentials before running the app
) else (
    echo [OK] .env already exists — skipping
)

REM ── 4. Install dependencies (UV auto-selects Python 3.11) ──────────────
echo.
echo [..] Installing dependencies with UV...
uv sync
if %errorlevel% neq 0 (
    echo [X] Dependency installation failed.
    exit /b 1
)

REM ── Done ────────────────────────────────────────────────────────────────
echo.
echo ===========================================
echo  [OK] Setup Complete
echo ===========================================
echo.
echo  Next steps:
echo.
echo    1. Edit .env with your credentials:
echo       - NESTLE_CLIENT_ID
echo       - NESTLE_CLIENT_SECRET
echo       - DATABRICKS_HOST
echo       - ARM_CLIENT_ID
echo       - ARM_CLIENT_SECRET
echo       - ARM_TENANT_ID
echo       - WAREHOUSE_ID
echo       - DATABRICKS_CATALOG  ^(optional, default: samples^)
echo       - DATABRICKS_SCHEMA   ^(optional, default: bakehouse^)
echo       - LANGFUSE_PUBLIC_KEY  (optional)
echo       - LANGFUSE_SECRET_KEY  (optional)
echo.
echo    2. Start the MCP server:
echo       uv run python -m agent_eval.mcp_server.server
echo.
echo    3. Launch the chat UI (separate terminal):
echo       uv run streamlit run agent_eval/ui/app.py
echo.
echo    4. Run evaluations (no MCP server needed):
echo       uv run python -m evaluation.runner
echo.
