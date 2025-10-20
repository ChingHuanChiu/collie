from typing import (
    Dict, 
    Any,
    Optional,
    List,
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


class TransformerPayload(BaseModel):
    train_data: Optional[pd.DataFrame] = None
    validation_data: Optional[pd.DataFrame]  = None
    test_data: Optional[pd.DataFrame]  = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TrainerPayload(BaseModel):
    model: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TunerPayload(BaseModel):
    hyperparameters: Dict[str, Any]
    train_data: Optional[pd.DataFrame] = None
    validation_data: Optional[pd.DataFrame]  = None
    test_data: Optional[pd.DataFrame]  = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EvaluatorPayload(BaseModel):
 
    metrics: List[Dict[str, Any]]
    is_better_than_production: bool


class PusherPayload(BaseModel):
    model_uri: str