# RAG Evaluation Cookbooks (Langfuse + RAGAS)

A collection of Jupyter notebook cookbooks demonstrating RAG evaluation patterns using **Langfuse** and **RAGAS**. These notebooks provide:

- End-to-end RAG evaluation workflows
- **RAGAS** metrics (Faithfulness, Answer Relevancy, Context Precision)
- **Langfuse** integration for tracing and experiment tracking
- **LangChain** callback support via `langfuse.langchain.CallbackHandler`

## Notebooks

| Notebook                                                                         | Description                                                                                                                                                                      |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [experiment_ragas_metrics.ipynb](cookbooks/experiment_ragas_metrics.ipynb)       | **Langfuse Experiments with RAGAS** — Run structured experiments using `langfuse_client.run_experiment()` with RAGAS metrics on the FIQA dataset. Ideal for CI/CD quality gates. |
| [trace_evaluation_with_ragas.ipynb](cookbooks/trace_evaluation_with_ragas.ipynb) | **Offline Trace Evaluation** — Retrieve existing Langfuse traces and evaluate them with RAGAS metrics after-the-fact. Useful for production trace analysis.                      |

## Quickstart

### Prerequisites

- Linux, macOS, or WSL on Windows
- Git or GitHub CLI (to clone this repository)
- Python 3.11+ (via [pyenv](https://github.com/pyenv/pyenv) recommended)
- [uv](https://github.com/astral-sh/uv) package manager
- VS Code (recommended for notebooks and debugging)

### One-time Computer Setup
These are things you need to do once on your workstation.

#### WSL Ubuntu - Trust Nestle Root Certificate
The Ubuntu image for WSL doesn't trust Nestle's root CA by default.
```bash
# Save the certificate
sudo tee /usr/local/share/ca-certificates/nestle-vpn.crt > /dev/null <<'EOF'
-----BEGIN CERTIFICATE-----
MIIHLjCCBRagAwIBAgIQNbnIIqTaa4NN48B84GXb5zANBgkqhkiG9w0BAQsFADAg
MR4wHAYDVQQDExVOZXN0bGUgQ1MgUFJEIFJvb3QgQ0EwHhcNMjMwMzA4MTM0NjE2
WhcNNDMwNjAxMDk1MDEzWjAgMR4wHAYDVQQDExVOZXN0bGUgQ1MgUFJEIFJvb3Qg
Q0EwggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQC5tO7bxOhA3krqYMNZ
Vc50jMREkLnoTsbNl0ajK7N5gMaDbozwF6P9OEtwnOpjP+/XVAp83IJ0UyhKGcb7
AiiT1yC3dMzfpojfEAvBonCFM0OCGeYwzHFDTTK5Yc+0yEGsdlGLacFKYDinAuUD
O+Wr7682buGZN5L8FET+ogdVV8EdKIMwxve5NVwjMHu5DCU/oivO2t9apH7DE5o9
GA0wg0cs3J1fZ7CFCPJkhYILBQacVj+/DHEGaQ/m4G1TvCrJ4hX+gBXVE+dzAEWF
Zw/9cvM7X5q6ki+XCZJwaW/Tn3KUQcYbKZES3IBXCtmEUXZYKr5DcTAsm7U9GhP8
kZ7g1Kh4c9zHiWomdMHjnLWyu2bvlpbSmz3yCli03uNoloM4xFWxKQ8dZGUrLVkO
klq3Mh2Xldu9XShEQPRxEElH3qNpzU5A1oWlWyJLxI0yWguFaMeNzisdFifIG9ZB
2SzJ7NRzW9IYWWkZO30kD0uS71xmYa3uWk9KMBJK7fzcCZeBo3XQmLWsP+DqCs0U
Ux7e3pNZNgv1R9ayc/w+vByQIcl4DBKOA09J2C9D7rBbF2WGFxnwMO5BjuHLnhQx
U/g8DtNdysePTylD1+PfooGQO5X+k9WzAA1TGwIjoAu1mmCYeyEiABBagC8tnTM9
6og0eOBdV8OhlDvhh2tEy6gRrQIDAQABo4ICYjCCAl4wDgYDVR0PAQH/BAQDAgEG
MBIGA1UdEwEB/wQIMAYBAf8CAQEwHQYDVR0OBBYEFD0UwM6FXl3tx8FZKwlt5uA2
bdbiMEAGA1UdHwQ5MDcwNaAzoDGGL2h0dHA6Ly9wa2ktY3JsLm5lc3RsZS5jb20v
TmVzdGxlQ1NQUkRSb290Q0EuY3JsMBAGCSsGAQQBgjcVAQQDAgEBMIIBUQYDVR0g
BIIBSDCCAUQwBgYEVR0gADCCATgGCSsGAQQBgtd9ATCCASkwQAYIKwYBBQUHAgEW
NGh0dHBzOi8vcGtpLWNybC5uZXN0bGUuY29tL0NQUy9OZXN0bGVJbnRlcm5hbENQ
Uy5wZGYwgeQGCCsGAQUFBwICMIHXHoHUAEMAZQByAHQAaQBmAGkAYwBhAHQAZQAg
AFAAbwBsAGkAYwB5ACAAYQBuAGQAIABDAGUAcgB0AGkAZgBpAGMAYQB0AGkAbwBu
ACAAUAByAGEAYwB0AGkAYwBlACAAUwB0AGEAdABlAG0AZQBuAHQAIABvAGYAIABO
AGUAcwB0AGwAZQAgAGUAbgB2AGkAcgBvAG4AbQBlAG4AdAAuAFUAbgBhAHUAdABo
AG8AcgBpAHoAZQBkACAAdQBzAGUAIABwAHIAbwBoAGkAYgBpAHQAZQBkAC4wSwYI
KwYBBQUHAQEEPzA9MDsGCCsGAQUFBzAChi9odHRwOi8vcGtpLWNybC5uZXN0bGUu
Y29tL05lc3RsZUNTUFJEUm9vdENBLmNydDAjBgkrBgEEAYI3FQIEFgQUh95Lze/u
mcAlJ6ebJc3b2Lq7ab4wDQYJKoZIhvcNAQELBQADggIBAIWkHpqom1XirwSSD/yJ
nRULOMl227cGZdK0/U0R85D9y3/Flf4HNWYWZ/lV1RQTBgkhn03919fydZrhwFJw
BH00xywSP8aXuVHQ8xgs/o/a3mc8qqPsZ5DDZp4Nzof+ysqDU6UMEHVUw8/JIioU
IYDWDzUqVE5hhN/jfXhn+cLlFRSJB5fdzroplBKSjojoF26h66BVuV1hu8UwTn3m
mhGW177HZVvEmOxLTkQZZMk1QkIfrWOQA9BOIgfDyV+OCj5syoK63Lr051kQzYx8
kb3Dj3HzhNUPJiaqG1WMCaR7xlSDieHKAPlM7hqKrsRAZ0dKd65FqnP+Ir7jq/rs
ewf0tD+8M/tu62mvEFBUQ7JWeJvcI83g/G/BQJ8bhYFUgQ8JbvOncf4kHqWQ1oAu
LAYnCA6YMKn7Zm3JxoqiW6XqfeP1rb6JMmCEHopKNQ3jTu1UqVVZm0p1X4twIGKF
iemKBk2jXylDguCb4P+7LS4ITqMCiFqFF87ZAIe6RFiwhm5/alA9IhrhPF2kv2uQ
Hep3X9ttHdlmZ8htCL+itNJBKiu54SP6eUew/xg0GS9WUM9fhg2B5wnerkq25s7G
3wvmSViAapG91h1yEVaRZoQXy8plvhHeX9UWZp7x2by793RdsHQTQg7LHWdhuxLl
vr7/Avftx90M1yXsiY0QcIwX
-----END CERTIFICATE-----
EOF

# Update certificate trust store
sudo update-ca-certificates
```

#### Install Python Build Dependencies
pyenv compiles Python from source. Without `libssl-dev`, the compiled Python may use a bundled OpenSSL that ignores the OS certificate store (including the Nestle root CA installed above).
```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev libffi-dev liblzma-dev libncurses-dev
```

#### Install pyenv and uv
```bash
# Install pyenv (if not already installed)
curl https://pyenv.run | bash

# Add pyenv to your shell (add to ~/.bashrc or ~/.zshrc)
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Tell Python and libraries (requests, httpx) to use the OS certificate bundle
# instead of certifi's embedded CA bundle. This ensures the Nestle root CA is trusted.
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Restart your shell or run:
exec "$SHELL"
```

#### Install Python 3.12
```bash
pyenv install 3.12
pyenv global 3.12

# Verify Python is linked to system OpenSSL (paths should reference /etc/ssl)
python -c "import ssl; print(ssl.get_default_verify_paths())"
```

#### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### First Time Project Setup
Do these things from the root directory of the repo that you've cloned to your local system. 

```bash
# Set Python version and create virtual environment
pyenv local 3.12
uv venv
source .venv/bin/activate

# Install dependencies with notebook support
uv sync --extra notebook

# Configure environment variables
cp .env.example .env  # fill in your API keys
```
When you open a new terminal in VS Code, it will automatically initialize this virtual environment.

## External Services

| Service      | Required | Setup Guide                           | Description                             |
| ------------ | -------- | ------------------------------------- | --------------------------------------- |
| **Langfuse** | ✅ Yes    | [Setup Guide](docs/setup-langfuse.md) | Observability, experiments, and scoring |
| **OpenAI**   | ✅ Yes    | [Setup Guide](docs/setup-openai.md)   | LLM for generation and RAGAS evaluation |

## VS Code Integration

### Recommended Extensions

This repository includes recommended VS Code extensions in `.vscode/extensions.json`. When you open the project, VS Code will prompt you to install them.

### Context7 MCP Server

The repository includes a [Context7 MCP](https://mcp.context7.com/) configuration for enhanced Copilot assistance with up-to-date library documentation.

**Setup:**

1. Open VS Code in this workspace
1. The MCP configuration is in `.vscode/mcp.json`
1. Start the MCP server:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
   - Type "MCP: List Servers" and select it
   - Click "Start" next to the Context7 server
1. When prompted, enter your Context7 API key (cached securely for future sessions)

**Usage:**

- Context7 provides Copilot with current documentation for libraries like LangChain, RAGAS, and Langfuse
- Use `/context7` in Copilot Chat to explicitly query library documentation

### Debug Configurations

Pre-configured debug settings are available in `.vscode/launch.json`:

- **Python: Current File** — Debug the active Python file
- **Python Debugger: Jupyter** — Debug Jupyter notebooks
- **Attach to Kernel** — Attach to a running Jupyter kernel

## License

MIT
