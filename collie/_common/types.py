from typing import (
    Dict, 
    Any,
    Optional
)
from enum import Enum, auto

from pydantic import BaseModel


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