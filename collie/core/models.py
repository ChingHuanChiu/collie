from typing import (
    Dict, 
    Any,
    Optional,
    Literal,
)
from pydantic import BaseModel, ConfigDict

import pandas as pd


class TransformerArtifactPath(BaseModel):
    train_data: str = "Transformer/train"
    validation_data: str = "Transformer/validation"
    test_data: str = "Transformer/test"


class TunerArtifactPath(BaseModel):
    hyperparameters: str = "Tuner/hyperparameters"


class TrainerArtifactPath(BaseModel):
    model: str = "Trainer/model"


class EvaluatorArtifactPath(BaseModel):
    metrics: str = "Evaluator/metrics"


class PusherArtifactPath(BaseModel):
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