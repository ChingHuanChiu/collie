from typing import (
    Dict, 
    Any,
    Optional,
    List,
)
from pydantic import BaseModel, ConfigDict, Field

import pandas as pd


class BasePayload(BaseModel):
    """Base class for all payload types with optional extra_data functionality."""
    
    extra_data: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def set_extra(self, key: str, value: Any):
        """
        Set a value in extra_data and return self for chaining.
        
        Args:
            key: The key to set
            value: The value to store
            
        Returns:
            Self for method chaining
            
        Example:
            >>> payload.set_extra("feature_names", ["age", "income"])
            >>> payload.set_extra("n_classes", 3).set_extra("version", "1.0")
        """
        if self.extra_data is None:
            self.extra_data = {}
        self.extra_data[key] = value
        return self
    
    def get_extra(self, key: str, default: Any = None) -> Any:
        """
        Get a value from extra_data with optional default.
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The value or default
            
        Example:
            >>> features = payload.get_extra("feature_names", [])
        """
        if self.extra_data is None:
            return default
        return self.extra_data.get(key, default)
    
    def has_extra(self, key: str) -> bool:
        """
        Check if a key exists in extra_data.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
            
        Example:
            >>> if payload.has_extra("feature_names"):
            ...     features = payload.get_extra("feature_names")
        """
        if self.extra_data is None:
            return False
        return key in self.extra_data


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


class TransformerPayload(BasePayload):
    train_data: Optional[pd.DataFrame] = None
    validation_data: Optional[pd.DataFrame]  = None
    test_data: Optional[pd.DataFrame]  = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TrainerPayload(BasePayload):
    model: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TunerPayload(BasePayload):
    hyperparameters: Dict[str, Any]
    train_data: Optional[pd.DataFrame] = None
    validation_data: Optional[pd.DataFrame]  = None
    test_data: Optional[pd.DataFrame]  = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EvaluatorPayload(BasePayload):
    metrics: List[Dict[str, Any]]
    is_better_than_production: bool


class PusherPayload(BasePayload):
    model_uri: str
    status: Optional[str] = None
    model_version: Optional[str] = None