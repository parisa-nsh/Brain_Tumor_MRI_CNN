import pytest
import os
import sys
from pathlib import Path

# Add src to Python path for importing modules
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Create and return a temporary test data directory"""
    test_data = project_root / "tests" / "test_data"
    test_data.mkdir(parents=True, exist_ok=True)
    return test_data 