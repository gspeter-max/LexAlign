# Docker Development Environment Design Document

**Date:** 2026-02-26
**Status:** Approved

## Overview

Create a Docker-based development environment that provides a reproducible Linux environment with modern torch (2.6+) for running all LexAlign CLIs (download, finetune, align) and E2E integration tests. This solves Mac compatibility issues with newer torch versions and provides isolated, reproducible test execution.

## Architecture

### Container Stack

**Base image:** `python:3.11-slim` (Debian-based, ~100MB)

**Layers:**
1. System dependencies (curl, git, build-essential)
2. Python dependencies (torch>=2.6, transformers, trl, peft, datasets, accelerate)
3. CLI dependencies (click, pyyaml, rich)
4. Test dependencies (pytest, pytest-cov, pytest-mock, pytest-timeout)
5. Project source (mounted volume, editable install)
6. Entry point (flexible script)

### Container Capabilities

- Run CLIs: `download.py`, `finetune.py`, `align.py`
- Run tests: `pytest` (all tests including integration)
- Interactive shell: `bash` for development
- Volume mount: Project directory for live code changes
- Environment variables: HF_TOKEN passthrough for Hugging Face access

## Components

### Files to Create

**Dockerfile** - Container definition
```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY lexalign/ ./lexalign/
COPY download.py finetune.py align.py ./
COPY config/ ./config/

# Install dependencies (editable mode for development)
RUN pip install --no-cache-dir -e .

# Copy entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
```

**docker-entrypoint.sh** - Flexible execution
```bash
#!/bin/bash
set -e

# Warn if HF_TOKEN not set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Hugging Face operations may fail."
    echo "Set with: docker run -e HF_TOKEN=your_token ..."
fi

# Execute command or default to bash
if [ $# -eq 0 ]; then
    exec "/bin/bash"
else
    exec "$@"
fi
```

**.dockerignore** - Exclude unnecessary files
```
.git
.gitignore
__pycache__
*.pyc
.venv
.eggs
*.egg-info
.pytest_cache
.coverage
.coverage.*
.tox
docs/
tests/tmp/
```

**docker-compose.yml** (Optional) - Convenient wrapper
```yaml
version: '3.8'

services:
  lexalign:
    build: .
    volumes:
      - .:/app
    environment:
      - HF_TOKEN=${HF_TOKEN}
    working_dir: /app
```

## Data Flow

### Interactive Development
```
User: docker run -it lexalign-dev bash
  ↓
Container starts with bash shell
  ↓
Source code mounted at /app (editable install)
  ↓
Changes reflected immediately
  ↓
Can run: python download.py, pytest, etc.
```

### Direct Command Execution
```
User: docker run lexalign-dev pytest tests/ -v
  ↓
Entrypoint receives command
  ↓
Executes pytest in container
  ↓
Returns exit code to host
```

### CI/CD Pipeline
```
CI: docker run lexalign-dev pytest tests/ -m integration
  ↓
Fresh environment each run
  ↓
Isolated from host dependencies
  ↓
Reproducible test results
```

## Error Handling

### Container Build Failures
- Clear error messages for missing system dependencies
- Pin torch version to avoid API breaking changes
- Verify pip installations with `pip check`

### Runtime Failures
- HF_TOKEN validation in entrypoint (warn if missing)
- Graceful handling of volume mount issues
- Clear error messages if CLIs fail

### Test Execution Failures
- Preserve test artifacts in mounted volume
- Capture logs to host filesystem
- Proper exit code propagation for CI/CD

### Common Issues

| Issue | Prevention/Solution |
|-------|---------------------|
| Mismatched torch versions | Pin versions in Dockerfile |
| Missing model cache | Volume mount for HF cache |
| Network issues for HF downloads | Timeout and retry in CLIs |
| Permission issues on mounted volumes | Run with appropriate user ID |

## Dependencies & Version Pinning

### Core ML Packages
```
torch==2.6.0
transformers==4.56.2
trl==0.28.0
peft==0.18.1
datasets==3.5.0
accelerate==1.5.2
bitsandbytes==0.45.2
```

### CLI & Testing Packages
```
click>=8.0
pyyaml>=6.0
rich>=13.0
pytest>=8.0
pytest-cov>=4.0
pytest-mock>=3.10
pytest-timeout>=2.0
```

### Hugging Face
```
huggingface-hub>=0.30.0
```

These versions are tested together and compatible. Update `pyproject.toml` with these pins.

## Testing Strategy

### Test the Docker Setup

1. **Smoke test** - Container builds successfully
2. **CLI test** - Can run CLIs with `--help`
3. **Unit test** - Fast tests pass in container
4. **Integration test** - E2E tests pass with real model

### Manual Testing Commands

```bash
# Build image
docker build -t lexalign-dev .

# Test CLIs work
docker run --rm lexalign-dev python download.py --help
docker run --rm lexalign-dev python finetune.py --help
docker run --rm lexalign-dev python align.py --help

# Run unit tests (fast, no network)
docker run --rm lexalign-dev pytest tests/ -v -m "not integration"

# Run integration tests (requires HF_TOKEN)
docker run --rm -e HF_TOKEN=$HF_TOKEN \
  lexalign-dev pytest tests/test_real_e2e.py -v -m integration

# Interactive shell for development
docker run -it -v $(pwd):/app lexalign-dev bash

# Run with docker-compose
docker-compose run --rm lexalign pytest tests/ -v
```

### CI/CD Integration

```yaml
# .github/workflows/docker-tests.yml
name: Docker Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t lexalign-dev .
      - name: Run unit tests
        run: docker run lexalign-dev pytest tests/ -v -m "not integration"
      - name: Run integration tests
        if: github.event_name == 'workflow_dispatch'
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: docker run lexalign-dev pytest tests/test_real_e2e.py -v -m integration
```

## Usage Examples

### Interactive Development

```bash
# Build the image
docker build -t lexalign-dev .

# Run with interactive shell
docker run -it -v $(pwd):/app lexalign-dev bash

# Inside container, you can:
python download.py --config config/downloads.yaml --dry-run
python finetune.py --config config/finetune.yaml --dry-run
python align.py --config config/align.yaml --dry-run
pytest tests/test_real_e2e.py::test_download_distilgpt2 -v -s
```

### Direct Command Execution

```bash
# Run all unit tests
docker run --rm -v $(pwd):/app lexalign-dev pytest tests/ -v -m "not integration"

# Run specific integration test
docker run --rm -v $(pwd):/app -e HF_TOKEN=$HF_TOKEN \
  lexalign-dev pytest tests/test_real_e2e.py::test_download_distilgpt2 -v

# Download model
docker run --rm -v $(pwd):/app -e HF_TOKEN=$HF_TOKEN \
  lexalign-dev python download.py --config config/downloads.yaml

# Fine-tune model
docker run --rm -v $(pwd):/app lexalign-dev \
  python finetune.py --config config/finetune.yaml
```

### Using Docker Compose (Optional)

```bash
# Set environment variable
export HF_TOKEN="your_token_here"

# Run tests
docker-compose run --rm lexalign pytest tests/ -v

# Run integration tests
docker-compose run --rm lexalign \
  pytest tests/test_real_e2e.py -v -m integration

# Interactive shell
docker-compose run --rm lexalign bash
```

## File Structure

```
LexAlign/
├── Dockerfile                      # NEW: Container definition
├── docker-entrypoint.sh            # NEW: Flexible entrypoint script
├── .dockerignore                   # NEW: Exclude patterns for build context
├── docker-compose.yml              # NEW: Optional convenience wrapper
├── pyproject.toml                  # UPDATE: Add pinned dependency versions
├── README.md                       # UPDATE: Add Docker usage section
├── download.py                     # EXISTING: No changes
├── finetune.py                     # EXISTING: No changes
├── align.py                        # EXISTING: No changes
├── lexalign/                       # EXISTING: No changes
├── config/                         # EXISTING: No changes
└── docs/
    └── plans/
        └── 2026-02-26-docker-dev-env-design.md  # NEW: This design document
```

## Success Criteria

- [ ] Docker image builds successfully
- [ ] All CLIs work in container (download, finetune, align)
- [ ] Unit tests pass in container
- [ ] Integration tests pass in container with real model download
- [ ] Volume mounting works for live code changes
- [ ] HF_TOKEN passthrough works
- [ ] Can use interactively (bash shell)
- [ ] Can execute commands directly (pytest, etc.)
- [ ] Documentation updated in README.md
- [ ] CI/CD can run tests in Docker

## Rationale

### Why Docker?

| Problem | Docker Solution |
|---------|----------------|
| Mac torch 2.2.2 incompatible with TRL 0.28.0 | Use Linux container with torch 2.6.0 |
| Dependency conflicts between projects | Isolated environment per project |
| "Works on my machine" issues | Reproducible builds |
| Hard to test integration scenarios | Fresh environment each run |
| CI/CD environment differences | Same container locally and in CI |

### Why Full Development Environment?

Instead of test-only container:
- Can develop and debug interactively
- Test CLIs directly, not just through pytest
- Flexible for various workflows
- Single environment for all tasks

### Why Editable Install?

- Volume mount changes reflected immediately
- No rebuild needed for code changes
- Faster iteration during development
- Maintains project structure

## Implementation Notes

### Performance Considerations

- Image size: ~2-3 GB (mostly torch)
- Build time: ~3-5 minutes first time, ~30 seconds cached
- Container startup: ~1-2 seconds
- Volume mount performance: Good on Mac Docker Desktop

### Security Considerations

- Don't commit HF_TOKEN to repo
- Use `.dockerignore` to exclude secrets
- Run as non-root user if needed (add to Dockerfile)
- Scan images for vulnerabilities if deploying

### Platform-Specific Notes

**Mac (Apple Silicon):**
- Docker Desktop supports both Intel and ARM
- torch builds available for both architectures
- May need `--platform linux/amd64` for some packages

**Linux:**
- Native Docker, no virtualization overhead
- Best performance
- GPU passthrough possible with nvidia-docker

**Windows:**
- WSL2 recommended for better performance
- Same commands as Mac/Linux
