"""
Comprehensive test suite for the collie ML pipeline framework.

This test suite covers all major components:
- Core models and data structures
- Component classes (Transformer, Trainer, Tuner, Evaluator, Pusher)
- Orchestrator
- Event system
- Enumerations
- Exception handling
- Utility functions

To run all tests:
    pytest tests/

To run specific test files:
    pytest tests/test_models.py
    pytest tests/test_transformer.py -v
    pytest tests/test_trainer.py::TestTrainer::test_run_success

To run with coverage:
    pytest tests/ --cov=collie --cov-report=html

To run tests in parallel:
    pytest tests/ -n auto
"""

import pytest
import sys
import os

# Add the collie package to the Python path for testing
# This allows tests to import from collie even if it's not installed
test_dir = os.path.dirname(os.path.abspath(__file__))
collie_dir = os.path.dirname(test_dir)
if collie_dir not in sys.path:
    sys.path.insert(0, collie_dir)


# Custom markers for organizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "component: mark test as a component test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_mlflow: mark test as requiring MLflow"
    )


# Test fixtures that can be used across multiple test files
@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample pandas DataFrame for testing."""
    import pandas as pd
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_context():
    """Fixture providing a sample PipelineContext for testing."""
    from collie.contracts.event import PipelineContext
    context = PipelineContext()
    context.set("model_uri", "runs:/123456/model")
    context.set("train_data_uri", "path/to/train.csv")
    context.set("test_run", True)
    return context


@pytest.fixture
def mock_mlflow_run():
    """Fixture providing a mock MLflow run for testing."""
    from unittest.mock import Mock
    run = Mock()
    run.info.run_id = "test-run-id-123"
    run.info.artifact_uri = "test://artifacts"
    run.info.experiment_id = "0"
    return run
