from typing import Any

import mlflow.sklearn
import sklearn.base
import mlflow.xgboost
import xgboost as xgb
import mlflow.pytorch
import torch.nn as nn
import mlflow.lightgbm
import lightgbm as lgb
import mlflow.transformers
from transformers import PreTrainedModel

from collie._common.mlflow_model_io.base_flavor_handler import FlavorHandler


class SklearnFlavorHandler(FlavorHandler):
    def can_handle(self, model: Any) -> bool:
        return isinstance(model, sklearn.base.BaseEstimator)

    def flavor(self) -> str:
        return "sklearn"

    def log_model(self, model: Any, name: str, **kwargs: Any) -> None:
        mlflow.sklearn.log_model(model, name, **kwargs)

    def load_model(self, model_uri: str) -> Any:
        return mlflow.sklearn.load_model(model_uri)
    

class XGBoostFlavorHandler(FlavorHandler):
    def can_handle(self, model: Any) -> bool:
        return isinstance(model, (xgb.Booster, xgb.XGBModel))

    def flavor(self) -> str:
        return "xgboost"

    def log_model(self, model: Any, name: str, **kwargs: Any) -> None:
        mlflow.xgboost.log_model(model, name, **kwargs)

    def load_model(self, model_uri: str) -> Any:
        return mlflow.xgboost.load_model(model_uri)



class PyTorchFlavorHandler(FlavorHandler):
    def can_handle(self, model: Any) -> bool:
        return isinstance(model, nn.Module)

    def flavor(self) -> str:
        return "pytorch"

    def log_model(self, model: Any, name: str, **kwargs: Any) -> None:
        mlflow.pytorch.log_model(model, name, **kwargs)

    def load_model(self, model_uri: str) -> Any:
        return mlflow.pytorch.load_model(model_uri)
    

class LightGBMFlavorHandler(FlavorHandler):
    def can_handle(self, model: Any) -> bool:
        return isinstance(model, (lgb.Booster, lgb.LGBMModel))

    def flavor(self) -> str:
        return "lightgbm"

    def log_model(self, model: Any, name: str, **kwargs: Any) -> None:
        mlflow.lightgbm.log_model(model, name, **kwargs)

    def load_model(self, model_uri: str) -> Any:
        return mlflow.lightgbm.load_model(model_uri)
    

class TransformersFlavorHandler(FlavorHandler):
    def can_handle(self, model: Any) -> bool:
        return isinstance(model, PreTrainedModel)

    def flavor(self) -> str:
        return "transformers"

    def log_model(self, model: Any, name: str, **kwargs: Any) -> None:
        mlflow.transformers.log_model(model, name, **kwargs)

    def load_model(self, model_uri: str) -> Any:
        return mlflow.transformers.load_model(model_uri)