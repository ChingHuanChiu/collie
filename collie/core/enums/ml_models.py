from enum import Enum


class ModelFlavor(str, Enum):
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    PYTORCH = "pytorch"
    TRANSFORMERS = "transformers"


class MLflowModelStage(str, Enum):
    
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
    
    def __str__(self) -> str:
        return self.value