# conftest.py
"""
Pytest configuration for LexAlign.

GPU tests are skipped by default because CI and development machines
typically lack a CUDA GPU.  Pass ``--run-gpu`` to include them:

    pytest tests/ --run-gpu

"""
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests that require a CUDA GPU (skipped by default).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-skip tests marked with @pytest.mark.gpu unless --run-gpu is passed."""
    if config.getoption("--run-gpu"):
        return  # user opted in — run everything

    skip_gpu = pytest.mark.skip(
        reason="GPU test skipped — pass --run-gpu to enable"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
