from typing import (
    Dict, 
    Any,
    Optional,
    Literal,
)
from pydantic import BaseModel, ConfigDict

import pandas as pd


class TransformerArtifact(BaseModel):
    train_data: str = "train.csv"
    validation_data: str = "validation.csv"
    test_data: str = "test.csv"


class TunerArtifact(BaseModel):
    hyperparameters: str = "hyperparameters.json"


class TrainerArtifact(BaseModel):
    model: str = "model"


class EvaluatorArtifact(BaseModel):
    report: str = "report.json"


class PusherArtifact(BaseModel):
    model_uri: str = "Pusher/model_uri"


class TransformerPayload(BaseModel):
    train_data: Optional[pd.DataFrame] = None
    validation_data: Optional[pd.DataFrame]  = None
    test_data: Optional[pd.DataFrame]  = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


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