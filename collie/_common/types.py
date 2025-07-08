from typing import (
    Dict, 
    Any,
    Optional,
    Literal
)
from enum import Enum, auto
from pydantic import BaseModel

import pandas as pd

from collie.transform.transform import Transformer
from collie.trainer.trainer import Trainer
from collie.tuner.tuner import Tuner
from collie.evaluator.evaluator import Evaluator
from collie.pusher.pusher import Pusher


class CollieComponents(Enum):

    TRAINER = Trainer
    TRANSFORMER = Transformer
    TUNER = Tuner 
    EVALUATOR = Evaluator
    PUSHER = Pusher


class EventType(Enum):
    INITAILIZE = auto()
    DATA_READY = auto()
    TRAINING_DONE = auto()
    TUNING_DONE = auto()
    EVALUATION_DONE = auto()
    PUSHER_DONE = auto()
    ERROR = auto()


class TransformerArtifactPath(Enum):
    train_data = "Transformer/train"
    validation_data = "Transformer/validation"
    test_data = "Transformer/test"


class TrainerArtifactPath(Enum):
    model = "Trainer/model"


class LocalArtifactDir(Enum):
    artifacts = "artifacts"

class TransformerPayload(BaseModel):
    train_data: Optional[pd.DataFrame] = None
    validation_data: Optional[pd.DataFrame]  = None
    test_data: Optional[pd.DataFrame]  = None


class TrainerPayload(BaseModel):
    model: Any = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None


class TunerPayload(BaseModel):
    hyperparameters: Dict[str, Any]


class EvaluatorPayload(BaseModel):
    metrics: Dict[Literal["Production", "Experiment"], Any]
    greater_is_better: bool = True


class PusherPayload(BaseModel):
    model_uri: str