"""Shared pytest fixtures and configuration."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SRC_DIR = PROJECT_ROOT / "src"

# Make src/ importable when running pytest without `pip install -e .`
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def raw_data_dir() -> Path:
    return DATA_DIR / "raw"


@pytest.fixture(scope="session")
def processed_data_dir() -> Path:
    return DATA_DIR / "processed"


@pytest.fixture(scope="session")
def outputs_dir() -> Path:
    return OUTPUTS_DIR


@pytest.fixture(scope="session")
def require_outputs(outputs_dir: Path) -> Path:
    """Skip integration test if outputs/ has not been populated (clean clone)."""
    canonical = outputs_dir / "etape5" / "predictions_test.csv"
    if not canonical.exists():
        pytest.skip(
            f"outputs/ not populated — missing {canonical.relative_to(PROJECT_ROOT)}. "
            f"Run the pipeline or restore the generated outputs/ artifacts first."
        )
    return outputs_dir
