from enum import Enum


class ModelFlavor(str, Enum):
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    PYTORCH = "pytorch"
    TRANSFORMERS = "transformers"