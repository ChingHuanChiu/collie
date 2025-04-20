from typing import (
    Dict, 
    Any,
    Optional
)
from enum import Enum, auto

from pydantic import BaseModel

from collie.transform.transform import Transformer
from collie.trainer.trainer import Trainer
from collie.tuner.tuner import Tuner
from collie.evaluator.evaluator import Evaluator


class CollieComponents(Enum):

    TRAINER = Trainer
    TRANSFORMER = Transformer
    TUNER = Tuner 
    EVALUATOR = Evaluator


class EventType(Enum):
    DATA_READY = auto()
    TRAINING_DONE = auto()
    TUNING_DONE = auto()
    EVALUATION_DONE = auto()
    ERROR = auto()


class TransformerPayload(BaseModel):
    data: Any


class TrainerPayload(BaseModel):
    model: Any
    train_loss: float
    val_loss: Optional[float]


class TunerPayload(BaseModel):
    hyperparameters: Dict[str, Any]


class EvaluatorPayload(BaseModel):
    metrics: Dict[str, Any]