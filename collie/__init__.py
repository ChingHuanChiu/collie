"""
Collie - A Lightweight MLOps Framework for Machine Learning Workflows

Collie provides a modular, event-driven architecture for building ML pipelines
with deep MLflow integration.

Quick Start:
    >>> from collie import Transformer, Trainer, Orchestrator
    >>> # Define your components
    >>> orchestrator = Orchestrator(
    ...     components=[MyTransformer(), MyTrainer()],
    ...     tracking_uri="http://localhost:5000",
    ...     registered_model_name="my_model"
    ... )
    >>> orchestrator.run()

For more examples, see: https://github.com/ChingHuanChiu/collie
"""

__author__ = "ChingHuanChiu"
__email__ = "stevenchiou8@gmail.com"
__version__ = "0.1.0b0"

# Import all main components for easy access
from .contracts.event import Event, EventType, PipelineContext
from .core.transform.transform import Transformer
from .core.trainer.trainer import Trainer
from .core.tuner.tuner import Tuner
from .core.evaluator.evaluator import Evaluator
from .core.pusher.pusher import Pusher
from .core.orchestrator.orchestrator import Orchestrator

# Import data models
from .core.models import (
    TransformerPayload,
    TrainerPayload,
    TunerPayload,
    EvaluatorPayload,
    PusherPayload,
)

# Import enums for configuration
from .core.enums.ml_models import ModelFlavor, MLflowModelStage

__all__ = [
    # Core components - the main classes users interact with
    "Transformer",
    "Trainer",
    "Tuner",
    "Evaluator",
    "Pusher",
    "Orchestrator",
    
    # Event system - for building custom workflows
    "Event",
    "EventType",
    "PipelineContext",
    
    # Payload models - for type-safe data passing
    "TransformerPayload",
    "TrainerPayload",
    "TunerPayload",
    "EvaluatorPayload",
    "PusherPayload",
    
    # Configuration enums
    "ModelFlavor",
    "MLflowModelStage",
]