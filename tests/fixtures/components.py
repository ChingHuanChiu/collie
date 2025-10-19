"""
Component fixtures for testing.

Provides pre-configured component instances for tests.
"""
import pytest
from collie.core.transform.transform import Transformer
from collie.core.trainer.trainer import Trainer
from collie.core.tuner.tuner import Tuner
from collie.core.evaluator.evaluator import Evaluator
from collie.core.pusher.pusher import Pusher
from collie.core.enums.ml_models import MLflowModelStage


@pytest.fixture
def transformer():
    """Provide a new Transformer instance for each test."""
    return Transformer(
        description="Test transformer",
        tags={"env": "test", "component": "Transformer"}
    )


@pytest.fixture
def trainer():
    """Provide a new Trainer instance for each test."""
    return Trainer(
        description="Test trainer",
        tags={"env": "test", "component": "Trainer"}
    )


@pytest.fixture
def tuner():
    """Provide a new Tuner instance for each test."""
    return Tuner(
        description="Test tuner",
        tags={"env": "test", "component": "Tuner"}
    )


@pytest.fixture
def evaluator():
    """Provide a new Evaluator instance for each test."""
    evaluator = Evaluator(
        target_stage=MLflowModelStage.STAGING,
        description="Test evaluator",
        tags={"env": "test", "component": "Evaluator"}
    )
    # Set required model name
    evaluator.registered_model_name = "test_model"
    return evaluator


@pytest.fixture
def pusher():
    """Provide a new Pusher instance for each test."""
    pusher = Pusher(
        target_stage=MLflowModelStage.PRODUCTION,
        archive_existing_versions=True,
        description="Test pusher",
        tags={"env": "test", "component": "Pusher"}
    )
    # Set required model name
    pusher.registered_model_name = "test_model"
    return pusher


@pytest.fixture
def all_components(transformer, tuner, trainer, evaluator, pusher):
    """Provide all components for full pipeline testing."""
    return [transformer, tuner, trainer, evaluator, pusher]


@pytest.fixture
def minimal_components(transformer, trainer):
    """Provide minimal components for simple pipeline testing."""
    return [transformer, trainer]
