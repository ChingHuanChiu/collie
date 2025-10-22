"""
Shared Pytest Configuration and Fixtures

This module contains pytest configuration, shared fixtures, and test utilities
that are available to all test files across the project.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collie.contracts.event import Event, EventType, PipelineContext
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
    config.addinivalue_line(
        "markers", "models: Tests for data models and schemas"
    )
    config.addinivalue_line(
        "markers", "events: Tests for event system"
    )
    config.addinivalue_line(
        "markers", "orchestrator: Tests for pipeline orchestration"
    )
    config.addinivalue_line(
        "markers", "mlflow: Tests requiring MLflow functionality"
    )
    config.addinivalue_line(
        "markers", "transformer: Tests for data transformation component"
    )
    config.addinivalue_line(
        "markers", "trainer: Tests for model training component"
    )
    config.addinivalue_line(
        "markers", "tuner: Tests for hyperparameter tuning component"
    )
    config.addinivalue_line(
        "markers", "evaluator: Tests for model evaluation component"
    )
    config.addinivalue_line(
        "markers", "pusher: Tests for model deployment component"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers based on location."""
    for item in items:
        # Auto-mark tests based on directory structure
        if "unit_tests" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        if "integration_tests" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark based on test file names
        if "test_event" in item.nodeid:
            item.add_marker(pytest.mark.events)
        if "test_orchestrator" in item.nodeid:
            item.add_marker(pytest.mark.orchestrator)
        if "test_transformer" in item.nodeid:
            item.add_marker(pytest.mark.transformer)
        if "test_trainer" in item.nodeid:
            item.add_marker(pytest.mark.trainer)
        if "test_tuner" in item.nodeid:
            item.add_marker(pytest.mark.tuner)
        if "test_evaluator" in item.nodeid:
            item.add_marker(pytest.mark.evaluator)
        if "test_pusher" in item.nodeid:
            item.add_marker(pytest.mark.pusher)


# ===== COMMON FIXTURES =====

@pytest.fixture
def sample_dataframe():
    """Provide a sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature3': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def sample_train_data():
    """Provide sample training data."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'target': [0, 1, 0, 1, 0, 1, 0, 1]
    })


@pytest.fixture
def sample_validation_data():
    """Provide sample validation data."""
    return pd.DataFrame({
        'feature1': [9, 10],
        'feature2': [0.9, 1.0],
        'target': [1, 0]
    })


@pytest.fixture
def sample_test_data():
    """Provide sample test data."""
    return pd.DataFrame({
        'feature1': [11, 12],
        'feature2': [1.1, 1.2],
        'target': [0, 1]
    })


@pytest.fixture
def pipeline_context():
    """Provide a fresh PipelineContext instance."""
    return PipelineContext()


@pytest.fixture
def pipeline_context_with_data():
    """Provide a PipelineContext with sample data."""
    context = PipelineContext()
    context.set("train_data_uri", "path/to/train.csv")
    context.set("model_uri", "runs:/123/model")
    context.set("experiment_name", "test_experiment")
    return context


@pytest.fixture
def transformer_payload(sample_train_data):
    """Provide a TransformerPayload instance with sample data."""
    return TransformerPayload(train_data=sample_train_data)


@pytest.fixture
def trainer_payload():
    """Provide a TrainerPayload instance."""
    mock_model = Mock()
    return TrainerPayload(
        model=mock_model,
        train_loss=0.5,
        val_loss=0.45
    )


@pytest.fixture
def evaluator_payload():
    """Provide an EvaluatorPayload instance."""
    return EvaluatorPayload(
        metrics=[
            {"accuracy": 0.95, "precision": 0.93},
            {"recall": 0.91, "f1": 0.92}
        ],
        is_better_than_production=True
    )


@pytest.fixture
def pusher_payload():
    """Provide a PusherPayload instance."""
    return PusherPayload(model_uri="runs:/123/model")


@pytest.fixture
def initialize_event(pipeline_context):
    """Provide an INITIALIZE event."""
    return Event(
        type=EventType.INITIALIZE,
        payload=None,
        context=pipeline_context
    )


@pytest.fixture
def data_ready_event(transformer_payload, pipeline_context):
    """Provide a DATA_READY event."""
    return Event(
        type=EventType.DATA_READY,
        payload=transformer_payload,
        context=pipeline_context
    )


@pytest.fixture
def training_done_event(trainer_payload, pipeline_context):
    """Provide a TRAINING_DONE event."""
    return Event(
        type=EventType.TRAINING_DONE,
        payload=trainer_payload,
        context=pipeline_context
    )


@pytest.fixture
def evaluation_done_event(evaluator_payload, pipeline_context):
    """Provide an EVALUATION_DONE event."""
    return Event(
        type=EventType.EVALUATION_DONE,
        payload=evaluator_payload,
        context=pipeline_context
    )


@pytest.fixture
def pusher_done_event(pusher_payload, pipeline_context):
    """Provide a PUSHER_DONE event."""
    return Event(
        type=EventType.PUSHER_DONE,
        payload=pusher_payload,
        context=pipeline_context
    )


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_mlflow_run():
    """Provide a mock MLflow run object."""
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-123"
    mock_run.info.experiment_id = "0"
    mock_run.info.artifact_uri = "test://artifacts"
    mock_run.info.status = "FINISHED"
    return mock_run


@pytest.fixture
def mock_component():
    """Provide a generic mock component."""
    component = Mock()
    component.run.return_value = Event(
        type=EventType.DATA_READY,
        payload=Mock(),
        context=PipelineContext()
    )
    return component


@pytest.fixture(autouse=True)
def reset_mlflow_tracking():
    """Reset MLflow tracking URI before each test to avoid interference."""
    import mlflow
    mlflow.set_tracking_uri("sqlite:///test_mlflow.db")
    yield
    # Cleanup after test
    try:
        mlflow.end_run()
    except:
        pass


# ===== HELPER FUNCTIONS =====

def assert_event_structure(event: Event, expected_type: EventType = None):
    """
    Assert that an event has the correct structure.
    
    Args:
        event: Event to validate
        expected_type: Expected EventType (optional)
    """
    assert isinstance(event, Event)
    assert isinstance(event.context, PipelineContext)
    if expected_type:
        assert event.type == expected_type


def create_mock_component_with_event(event_type: EventType, payload=None):
    """
    Create a mock component that returns a specific event.
    
    Args:
        event_type: Type of event to return
        payload: Payload for the event
        
    Returns:
        Mock component
    """
    component = Mock()
    component.run.return_value = Event(
        type=event_type,
        payload=payload if payload else Mock(),
        context=PipelineContext()
    )
    return component
