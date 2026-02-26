# Docker Development Environment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Docker-based development environment that provides reproducible Linux environment with torch 2.6+ for running all LexAlign CLIs and E2E integration tests.

**Architecture:** Multi-stage Dockerfile with python:3.11-slim base, editable project install, flexible entrypoint script for both interactive shell and direct command execution, volume mounts for live code changes.

**Tech Stack:** Docker, Docker Compose (optional), Python 3.11, torch 2.6.0, transformers 4.56.2, trl 0.28.0

---

### Task 1: Update pyproject.toml with Pinned Versions

**Files:**
- Modify: `pyproject.toml`

**Step 1: Read current pyproject.toml**

Run: `cat pyproject.toml`
Expected: See current dependencies (may be unpinned)

**Step 2: Add pinned versions to dependencies**

Add/update these versions in `pyproject.toml`:

```toml
[project]
name = "lexalign"
version = "0.1.0"
dependencies = [
    # Core ML packages (pinned for Docker reproducibility)
    "torch==2.6.0",
    "transformers==4.56.2",
    "trl==0.28.0",
    "peft==0.18.1",
    "datasets==3.5.0",
    "accelerate==1.5.2",
    "bitsandbytes==0.45.2",

    # Hugging Face
    "huggingface-hub>=0.30.0",

    # CLI & utilities
    "click>=8.0",
    "pyyaml>=6.0",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "pytest-mock>=3.10",
    "pytest-timeout>=2.0",
]
```

**Step 3: Verify pyproject.toml is valid**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`
Expected: No errors

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: pin dependency versions for Docker reproducibility"
```

---

### Task 2: Create .dockerignore File

**Files:**
- Create: `.dockerignore`

**Step 1: Write the .dockerignore file**

```bash
# Create .dockerignore
cat > .dockerignore << 'EOF'
# Git
.git
.gitignore
.github

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build
.eggs
lib
lib64
parts
sdist
var
wheels
pip-wheel-metadata
share/python-wheels
*.manifest
*.spec

# Virtual environments
.venv
venv/
ENV/
env/

# Testing
.pytest_cache
.coverage
.coverage.*
htmlcov/
.tox/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docs (don't need in container)
docs/
*.md
!README.md

# Tests artifacts
tests/tmp/
/tmp/

# CI/CD
.gitlab-ci.yml
.github/
EOF
```

**Step 2: Verify .dockerignore exists**

Run: `ls -la .dockerignore`
Expected: File exists with content

**Step 3: Commit**

```bash
git add .dockerignore
git commit -m "build: add .dockerignore for faster Docker builds"
```

---

### Task 3: Create Dockerfile

**Files:**
- Create: `Dockerfile`

**Step 1: Write the Dockerfile**

```dockerfile
# Use Python 3.11 slim image (Debian-based, ~100MB)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy pyproject.toml first (better caching)
COPY pyproject.toml ./

# Install project and dependencies (editable mode for development)
RUN pip install --no-cache-dir -e ".[dev]"

# Copy project files
COPY lexalign/ ./lexalign/
COPY download.py finetune.py align.py ./
COPY config/ ./config/
COPY tests/ ./tests/

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["bash"]
```

**Step 2: Verify Dockerfile syntax**

Run: `docker build --check -f Dockerfile .`
Expected: No syntax errors

**Step 3: Commit**

```bash
git add Dockerfile
git commit -m "build: add Dockerfile for development environment"
```

---

### Task 4: Create docker-entrypoint.sh Script

**Files:**
- Create: `docker-entrypoint.sh`

**Step 1: Write the entrypoint script**

```bash
#!/bin/bash
set -e

# Entrypoint script for LexAlign Docker container
# Supports both interactive shell and direct command execution

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Warning if HF_TOKEN not set
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set${NC}"
    echo "Hugging Face operations will fail without a token."
    echo "Set with: docker run -e HF_TOKEN=your_token ..."
    echo ""
fi

# Show LexAlign version/info if available
if [ -f "/app/lexalign/__init__.py" ]; then
    echo "LexAlign Docker Environment"
    echo "Working directory: /app"
    echo ""
fi

# Execute command or default to bash
if [ $# -eq 0 ]; then
    # No arguments: start interactive bash shell
    exec "/bin/bash"
else
    # Arguments provided: execute them
    exec "$@"
fi
```

**Step 2: Make script executable (if not on Windows)**

Run: `chmod +x docker-entrypoint.sh`
Expected: No errors

**Step 3: Verify script syntax**

Run: `bash -n docker-entrypoint.sh`
Expected: No syntax errors

**Step 4: Commit**

```bash
git add docker-entrypoint.sh
git commit -m "build: add flexible entrypoint script for Docker"
```

---

### Task 5: Create docker-compose.yml (Optional)

**Files:**
- Create: `docker-compose.yml`

**Step 1: Write the docker-compose.yml**

```yaml
version: '3.8'

services:
  lexalign:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      # Mount project directory for live code changes
      - .:/app
      # Preserve pip cache
      - pip-cache:/root/.cache/pip
      # Preserve HF model cache
      - hf-cache:/root/.cache/huggingface
    environment:
      # Pass HF_TOKEN from host environment
      - HF_TOKEN=${HF_TOKEN}
    working_dir: /app
    # Keep container running for interactive use
    stdin_open: true
    tty: true

volumes:
  pip-cache:
  hf-cache:
```

**Step 2: Verify docker-compose.yml syntax**

Run: `docker-compose config`
Expected: No syntax errors

**Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "build: add docker-compose.yml for convenience"
```

---

### Task 6: Build Docker Image

**Files:**
- None (build verification)

**Step 1: Build the Docker image**

Run: `docker build -t lexalign-dev .`
Expected: Build completes successfully (~3-5 minutes)

**Step 2: Verify image was created**

Run: `docker images | grep lexalign-dev`
Expected: Image listed with size

**Step 3: Tag image with version**

Run: `docker tag lexalign-dev lexalign-dev:0.1.0`
Expected: No errors

**Step 4: Test basic container functionality**

Run: `docker run --rm lexalign-dev python --version`
Expected: Shows Python 3.11.x

**Step 5: Commit (no files, just verification)**

```bash
# No commit needed, just verification step
echo "Docker image build verified"
```

---

### Task 7: Test CLIs Work in Container

**Files:**
- None (verification tests)

**Step 1: Test download.py CLI**

Run: `docker run --rm lexalign-dev python download.py --help`
Expected: Shows download.py help message

**Step 2: Test finetune.py CLI**

Run: `docker run --rm lexalign-dev python finetune.py --help`
Expected: Shows finetune.py help message

**Step 3: Test align.py CLI**

Run: `docker run --rm lexalign-dev python align.py --help`
Expected: Shows align.py help message

**Step 4: Verify pytest is available**

Run: `docker run --rm lexalign-dev pytest --version`
Expected: Shows pytest version

**Step 5: Document successful test**

No commit needed, verification complete.

---

### Task 8: Test Unit Tests in Container

**Files:**
- None (verification tests)

**Step 1: Run unit tests (fast, no network)**

Run: `docker run --rm -v $(pwd):/app lexalign-dev pytest tests/ -v -m "not integration" --tb=short`
Expected: Unit tests pass

**Step 2: Verify test count**

Run: `docker run --rm -v $(pwd):/app lexalign-dev pytest tests/ -m "not integration" --collect-only -q | tail -5`
Expected: Shows number of tests collected

**Step 3: Run with coverage**

Run: `docker run --rm -v $(pwd):/app lexalign-dev pytest tests/ -m "not integration" --cov=lexalign --cov-report=term`
Expected: Coverage report displayed

**Step 4: Document results**

No commit needed, verification complete.

---

### Task 9: Test Integration Tests in Container

**Files:**
- None (verification tests)

**Step 1: Run integration tests with HF_TOKEN**

Run: `docker run --rm -v $(pwd):/app -e HF_TOKEN=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 lexalign-dev pytest tests/test_real_e2e.py -v -m integration --tb=short`
Expected: Integration tests pass (downloads model, trains, etc.)

**Step 2: Verify test output**

Check that:
- Model download succeeds
- Fine-tuning completes
- DPO alignment completes
- Checkpoint resume works

**Step 3: Run specific test**

Run: `docker run --rm -v $(pwd):/app -e HF_TOKEN=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 lexalign-dev pytest tests/test_real_e2e.py::test_download_distilgpt2 -v`
Expected: Single test passes

**Step 4: Document results**

No commit needed, verification complete.

---

### Task 10: Test Interactive Shell

**Files:**
- None (verification tests)

**Step 1: Start interactive container**

Run: `docker run -it -v $(pwd):/app lexalign-dev bash`
Expected: Container starts with bash prompt inside

**Step 2: Inside container, run a command**

Run (inside container): `python download.py --version 2>&1 || echo "No version option"`
Expected: Command executes

**Step 3: Inside container, run pytest**

Run (inside container): `pytest tests/test_auth.py -v`
Expected: Tests run

**Step 4: Exit container**

Run (inside container): `exit`
Expected: Container exits, returns to host shell

**Step 5: Document results**

No commit needed, verification complete.

---

### Task 11: Test docker-compose (If created)

**Files:**
- None (verification tests)

**Step 1: Start container with docker-compose**

Run: `docker-compose run --rm lexalign bash`
Expected: Container starts with bash prompt

**Step 2: Test command execution**

Run: `docker-compose run --rm lexalign pytest tests/ -v -m "not integration" --maxfail=5`
Expected: Tests run in container

**Step 3: Test environment variable passthrough**

Run: `docker-compose run lexalign env | grep HF_TOKEN`
Expected: HF_TOKEN displayed (if set on host)

**Step 4: Clean up**

Run: `docker-compose down -v`
Expected: Volumes removed

**Step 5: Document results**

No commit needed, verification complete.

---

### Task 12: Update README.md with Docker Section

**Files:**
- Modify: `README.md`

**Step 1: Read current README**

Run: `head -100 README.md`
Expected: See current README structure

**Step 2: Add Docker section after Fine-Tuning section**

Add to README.md:

```markdown
## Docker Development Environment

A Docker environment is provided for reproducible development and testing, especially useful on Mac where torch version compatibility may be an issue.

### Building the Docker Image

```bash
docker build -t lexalign-dev .
```

### Running Tests in Docker

**Unit tests (fast, no network required):**
```bash
docker run --rm -v $(pwd):/app lexalign-dev pytest tests/ -v -m "not integration"
```

**Integration tests (requires HF_TOKEN):**
```bash
docker run --rm -v $(pwd):/app \
  -e HF_TOKEN=$HF_TOKEN \
  lexalign-dev pytest tests/test_real_e2e.py -v -m integration
```

### Using the CLIs in Docker

**Download models/datasets:**
```bash
docker run --rm -v $(pwd):/app \
  -e HF_TOKEN=$HF_TOKEN \
  lexalign-dev python download.py --config config/downloads.yaml
```

**Fine-tune models:**
```bash
docker run --rm -v $(pwd):/app \
  lexalign-dev python finetune.py --config config/finetune.yaml
```

**DPO alignment:**
```bash
docker run --rm -v $(pwd):/app \
  lexalign-dev python align.py --config config/align.yaml
```

### Interactive Development

```bash
# Start interactive shell
docker run -it -v $(pwd):/app lexalign-dev bash

# Inside container, you can:
# - Run pytest
# - Execute CLIs
# - Edit code (changes reflected via volume mount)
```

### Using Docker Compose (Optional)

```bash
# Run tests
docker-compose run --rm lexalign pytest tests/ -v

# Interactive shell
docker-compose run --rm lexalign bash
```

### Notes

- The Docker image uses Python 3.11 with torch 2.6.0
- Project is installed in editable mode for live code changes
- HF_TOKEN must be provided for Hugging Face operations
- First build takes 3-5 minutes, subsequent builds use cache
```

**Step 3: Verify README formatting**

Run: `cat README.md | tail -50`
Expected: Docker section visible at end

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add Docker development environment section to README"
```

---

### Task 13: Add CI/CD Configuration for Docker Tests

**Files:**
- Create: `.github/workflows/docker-tests.yml`

**Step 1: Create GitHub Actions workflow directory**

Run: `mkdir -p .github/workflows`
Expected: Directory created

**Step 2: Write the Docker test workflow**

```yaml
name: Docker Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:  # Allow manual trigger for integration tests

jobs:
  docker-unit-tests:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t lexalign-dev .

      - name: Run unit tests
        run: |
          docker run --rm lexalign-dev \
            pytest tests/ -v -m "not integration" --maxfail=10

  docker-integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch'  # Only run manually

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t lexalign-dev .

      - name: Run integration tests
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          docker run --rm \
            -e HF_TOKEN=$HF_TOKEN \
            lexalign-dev \
            pytest tests/test_real_e2e.py -v -m integration --timeout=600
```

**Step 3: Verify workflow YAML syntax**

Run: `cat .github/workflows/docker-tests.yml`
Expected: Valid YAML structure

**Step 4: Commit**

```bash
git add .github/workflows/docker-tests.yml
git commit -m "ci: add Docker test workflow to GitHub Actions"
```

---

### Task 14: Final Verification - Full Workflow

**Files:**
- None (final verification)

**Step 1: Clean build from scratch**

Run:
```bash
docker system prune -f
docker build --no-cache -t lexalign-dev .
```
Expected: Clean build completes

**Step 2: Run full test suite**

Run:
```bash
# Unit tests
docker run --rm -v $(pwd):/app lexalign-dev pytest tests/ -v -m "not integration"

# Integration tests (if HF_TOKEN available)
docker run --rm -v $(pwd):/app \
  -e HF_TOKEN=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 \
  lexalign-dev pytest tests/test_real_e2e.py -v -m integration
```
Expected: All tests pass

**Step 3: Test all CLIs work**

Run:
```bash
docker run --rm lexalign-dev python download.py --help
docker run --rm lexalign-dev python finetune.py --help
docker run --rm lexalign-dev python align.py --help
```
Expected: All help messages display

**Step 4: Verify documentation**

Run: `grep -A 5 "## Docker" README.md`
Expected: Docker section exists

**Step 5: Create summary of what was built**

No commit needed, final verification complete.

---

## Summary

This implementation plan creates a complete Docker development environment with:

1. **Reproducible builds** - Pinned dependency versions in pyproject.toml
2. **Fast iteration** - Volume mounts for live code changes
3. **Flexible execution** - Both interactive shell and direct commands
4. **Complete testing** - Unit and integration tests in container
5. **Documentation** - Updated README with Docker usage
6. **CI/CD integration** - GitHub Actions workflow for Docker tests

**Estimated time:** 45-60 minutes total (first build takes 3-5 minutes)

**Key benefits:**
- Solves Mac torch compatibility issues
- Isolated, reproducible test environment
- Same environment locally and in CI
- Easy onboarding for new developers
