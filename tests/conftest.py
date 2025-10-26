import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collie.contracts.event import Event, EventType, PipelineContext
from collie.contracts.mlflow import MLflowConfig
from collie.core.models import (
    TransformerPayload,
    TrainerPayload,
    TunerPayload,
    EvaluatorPayload,
    PusherPayload
)


# ===== PYTEST CONFIGURATION =====

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add any dynamic configuration here
    pass


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Auto-mark tests based on file names
        if "test_models" in item.nodeid:
            item.add_marker(pytest.mark.models)
        elif "test_event" in item.nodeid:
            item.add_marker(pytest.mark.events)
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif any(component in item.nodeid for component in 
                ["transformer", "trainer", "tuner", "evaluator", "pusher"]):
            item.add_marker(pytest.mark.component)
        
        # Auto-mark slow tests
        if any(indicator in item.name.lower() for indicator in 
               ["integration", "e2e", "full_pipeline", "slow"]):
            item.add_marker(pytest.mark.slow)
        
        # Auto-mark unit tests (default for most tests)
        if not any(marker.name in ["integration", "slow", "component"] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# ===== SHARED FIXTURES =====

@pytest.fixture
def sample_dataframe():
    """Provide a sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature3': ['A', 'B', 'C', 'D', 'E'],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def pipeline_context():
    """Provide a pre-configured PipelineContext for testing."""
    context = PipelineContext()
    context.set("test_run", True)
    context.set("experiment_id", "test_experiment")
    context.set("run_id", "test_run_123")
    return context


@pytest.fixture
def mock_mlflow_run():
    """Provide a mock MLflow run object."""
    run = Mock()
    run.info.run_id = "test-run-id-123"
    run.info.artifact_uri = "file:///tmp/test_artifacts"
    run.info.experiment_id = "0"
    run.info.status = "FINISHED"
    return run


@pytest.fixture
def mock_mlflow_config(tmp_path):
    """Provide a mock MLflow configuration for testing."""
    # Reset singleton for testing
    MLflowConfig._instance = None
    
    tracking_uri = f"sqlite:///{tmp_path}/test_mlflow.db"
    config = MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name="test_experiment"
    )
    config.configure()
    
    yield config
    
    # Reset singleton after test
    MLflowConfig._instance = None


@pytest.fixture
def transformer_payload(sample_dataframe):
    """Provide a sample TransformerPayload."""
    return TransformerPayload(
        train_data=sample_dataframe,
        validation_data=sample_dataframe.iloc[:3],
        test_data=sample_dataframe.iloc[3:]
    )


@pytest.fixture
def trainer_payload():
    """Provide a sample TrainerPayload."""
    mock_model = Mock()
    mock_model.predict.return_value = [0, 1, 0]
    
    return TrainerPayload(
        model=mock_model,
        train_loss=0.5,
        val_loss=0.3
    )


@pytest.fixture
def evaluator_payload():
    """Provide a sample EvaluatorPayload."""
    return EvaluatorPayload(
        metrics=[
            {"accuracy": 0.95},
            {"precision": 0.92},
            {"recall": 0.89}
        ],
        is_better_than_production=True
    )


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ===== TEST UTILITIES =====

@pytest.fixture
def assert_event_valid():
    """Provide a utility function to validate Event objects."""
    def _assert_event_valid(event, expected_type=None, expected_payload_type=None):
        """Assert that an event has the expected structure."""
        assert isinstance(event, Event)
        assert hasattr(event, 'type')
        assert hasattr(event, 'payload')
        assert hasattr(event, 'context')
        assert isinstance(event.context, PipelineContext)
        
        if expected_type:
            assert event.type == expected_type
        
        if expected_payload_type:
            assert isinstance(event.payload, expected_payload_type)
    
    return _assert_event_valid


# Skip conditions for optional dependencies
try:
    import mlflow
    mlflow_available = True
except ImportError:
    mlflow_available = False

skip_if_no_mlflow = pytest.mark.skipif(
    not mlflow_available, 
    reason="MLflow not available"
)

skip_if_ci = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skipping in CI environment"
)
import os

# Add the collie package to Python path
test_dir = os.path.dirname(os.path.abspath(__file__))
collie_dir = os.path.dirname(test_dir)
if collie_dir not in sys.path:
    sys.path.insert(0, collie_dir)


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "component: mark test as a component test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_mlflow: mark test as requiring MLflow")


# Shared fixtures
@pytest.fixture
def sample_train_data():
    """Fixture providing sample training data."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'feature3': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })


@pytest.fixture
def sample_validation_data():
    """Fixture providing sample validation data."""
    return pd.DataFrame({
        'feature1': [11, 12, 13],
        'feature2': [1.1, 1.2, 1.3],
        'feature3': [110, 120, 130],
        'target': [1, 0, 1]
    })


@pytest.fixture
def sample_test_data():
    """Fixture providing sample test data."""
    return pd.DataFrame({
        'feature1': [14, 15],
        'feature2': [1.4, 1.5],
        'feature3': [140, 150],
        'target': [0, 1]
    })


@pytest.fixture
def sample_context():
    """Fixture providing a sample PipelineContext."""
    from collie.contracts.event import PipelineContext
    context = PipelineContext()
    context.set("model_uri", "runs:/123456/model")
    context.set("train_data_uri", "path/to/train.csv")
    context.set("validation_data_uri", "path/to/validation.csv")
    context.set("test_data_uri", "path/to/test.csv")
    context.set("experiment_id", "test_experiment")
    return context


@pytest.fixture
def mock_mlflow_run():
    """Fixture providing a mock MLflow run."""
    run = Mock()
    run.info.run_id = "test-run-id-123456"
    run.info.artifact_uri = "test://artifacts/run123"
    run.info.experiment_id = "0"
    run.info.status = "FINISHED"
    return run


@pytest.fixture
def sample_hyperparameters():
    """Fixture providing sample hyperparameters."""
    return {
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 10,
        "dropout_rate": 0.2,
        "hidden_units": 128,
        "optimizer": "adam"
    }


@pytest.fixture
def sample_metrics():
    """Fixture providing sample evaluation metrics."""
    return [
        {"accuracy": 0.95, "precision": 0.92},
        {"recall": 0.89, "f1_score": 0.90},
        {"auc": 0.87, "loss": 0.12}
    ]


@pytest.fixture
def mock_model():
    """Fixture providing a mock ML model."""
    model = Mock()
    model.predict.return_value = [0, 1, 0, 1]
    model.score.return_value = 0.85
    model.__class__.__name__ = "MockModel"
    return model


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to most tests by default
        if not any(marker.name in ['integration', 'slow'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add component marker to component-specific tests
        component_names = ['transformer', 'trainer', 'tuner', 'evaluator', 'pusher', 'orchestrator']
        if any(component in item.nodeid.lower() for component in component_names):
            item.add_marker(pytest.mark.component)
        
        # Add slow marker to potentially slow tests
        slow_indicators = ['integration', 'full_workflow', 'pipeline', 'end_to_end']
        if any(indicator in item.name.lower() for indicator in slow_indicators):
            item.add_marker(pytest.mark.slow)


# Utility functions available to all tests
class TestHelpers:
    """Helper functions for tests."""
    
    @staticmethod
    def assert_event_structure(event):
        """Assert that an event has the expected structure."""
        from collie.contracts.event import Event, EventType, PipelineContext
        
        assert isinstance(event, Event)
        assert hasattr(event, 'type')
        assert hasattr(event, 'payload')
        assert hasattr(event, 'context')
        assert isinstance(event.context, PipelineContext)
    
    @staticmethod
    def assert_payload_structure(payload, expected_type):
        """Assert that a payload has the expected structure and type."""
        from pydantic import BaseModel
        
        assert isinstance(payload, expected_type)
        assert isinstance(payload, BaseModel)
        
        # Test that the payload can be serialized
        payload_dict = payload.model_dump()
        assert isinstance(payload_dict, dict)
    
    @staticmethod
    def create_mock_component(component_class):
        """Create a mock component with the basic interface."""
        mock = Mock(spec=component_class)
        mock.run.return_value = Mock()
        mock._registered_model_name = None
        return mock


# Make TestHelpers available as a fixture
@pytest.fixture
def test_helpers():
    """Fixture providing test helper functions."""
    return TestHelpers


# Performance testing fixture
@pytest.fixture
def performance_timer():
    """Fixture for timing test execution."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed_time
        
        @property
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Skip markers for missing dependencies
def pytest_runtest_setup(item):
    """Skip tests based on missing dependencies."""
    # Skip MLflow tests if MLflow is not available
    if item.get_closest_marker("requires_mlflow"):
        try:
            import mlflow
        except ImportError:
            pytest.skip("MLflow not available")


# Custom assertion helpers
def assert_dataframes_equal(df1, df2, check_dtype=True):
    """Assert that two DataFrames are equal."""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def assert_dict_subset(subset, superset):
    """Assert that one dict is a subset of another."""
    for key, value in subset.items():
        assert key in superset, f"Key '{key}' not found in superset"
        assert superset[key] == value, f"Value mismatch for key '{key}'"


# Make assertion helpers available as fixtures
@pytest.fixture
def assert_helpers():
    """Fixture providing assertion helper functions."""
    return {
        'assert_dataframes_equal': assert_dataframes_equal,
        'assert_dict_subset': assert_dict_subset
    }
