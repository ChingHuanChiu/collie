from typing import Any
import warnings


# Import with better error handling
try:
    import mlflow.sklearn
    import sklearn.base
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. SklearnFlavorHandler will be disabled.")

try:
    import mlflow.xgboost
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. XGBoostFlavorHandler will be disabled.")

try:
    import mlflow.pytorch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. PyTorchFlavorHandler will be disabled.")

try:
    import mlflow.lightgbm
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception):
    LIGHTGBM_AVAILABLE = False
    lgb = None
    warnings.warn("LightGBM not available. LightGBMFlavorHandler will be disabled.")

try:
    import mlflow.transformers
    from transformers import PreTrainedModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. TransformersFlavorHandler will be disabled.")

from collie._common.mlflow_model_io.base_flavor_handler import FlavorHandler
from collie.core.enums.ml_models import ModelFlavor
from collie._common.exceptions import ModelFlavorError


class SklearnFlavorHandler(FlavorHandler):
    """Handler for scikit-learn models."""
    
    def can_handle(self, model: Any) -> bool:
        if not SKLEARN_AVAILABLE:
            return False
        return isinstance(model, sklearn.base.BaseEstimator)

    def flavor(self) -> ModelFlavor:
        return ModelFlavor.SKLEARN

    def log_model(self, model: Any, name: str, **kwargs: Any) -> None:
        if not SKLEARN_AVAILABLE:
            raise ModelFlavorError(
                "scikit-learn is not available. Please install it to log sklearn models.",
                flavor="sklearn"
            )
        try:
            mlflow.sklearn.log_model(model, name, **kwargs)
        except Exception as e:
            raise ModelFlavorError(
                f"Failed to log sklearn model: {str(e)}",
                flavor="sklearn",
                details={"model_type": type(model).__name__, "artifact_name": name}
            ) from e

    def load_model(self, model_uri: str) -> Any:
        if not SKLEARN_AVAILABLE:
            raise ModelFlavorError(
                "scikit-learn is not available. Please install it to load sklearn models.",
                flavor="sklearn"
            )
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            raise ModelFlavorError(
                f"Failed to load sklearn model: {str(e)}",
                flavor="sklearn",
                details={"model_uri": model_uri}
            ) from e
    

class XGBoostFlavorHandler(FlavorHandler):
    """Handler for XGBoost models."""
    
    def can_handle(self, model: Any) -> bool:
        if not XGBOOST_AVAILABLE:
            return False
        return isinstance(model, (xgb.Booster, xgb.XGBModel))

    def flavor(self) -> ModelFlavor:
        return ModelFlavor.XGBOOST

    def log_model(self, model: Any, name: str, **kwargs: Any) -> None:
        if not XGBOOST_AVAILABLE:
            raise ModelFlavorError(
                "XGBoost is not available. Please install it to log XGBoost models.",
                flavor="xgboost"
            )
        try:
            mlflow.xgboost.log_model(model, name, **kwargs)
        except Exception as e:
            raise ModelFlavorError(
                f"Failed to log XGBoost model: {str(e)}",
                flavor="xgboost",
                details={"model_type": type(model).__name__, "artifact_name": name}
            ) from e

    def load_model(self, model_uri: str) -> Any:
        if not XGBOOST_AVAILABLE:
            raise ModelFlavorError(
                "XGBoost is not available. Please install it to load XGBoost models.",
                flavor="xgboost"
            )
        try:
            return mlflow.xgboost.load_model(model_uri)
        except Exception as e:
            raise ModelFlavorError(
                f"Failed to load XGBoost model: {str(e)}",
                flavor="xgboost",
                details={"model_uri": model_uri}
            ) from e


class PyTorchFlavorHandler(FlavorHandler):
    """Handler for PyTorch models."""
    
    def can_handle(self, model: Any) -> bool:
        if not PYTORCH_AVAILABLE:
            return False
        return isinstance(model, nn.Module)

    def flavor(self) -> ModelFlavor:
        return ModelFlavor.PYTORCH

    def log_model(self, model: Any, name: str, **kwargs: Any) -> None:
        if not PYTORCH_AVAILABLE:
            raise ModelFlavorError(
                "PyTorch is not available. Please install it to log PyTorch models.",
                flavor="pytorch"
            )
        try:
            mlflow.pytorch.log_model(model, name, **kwargs)
        except Exception as e:
            raise ModelFlavorError(
                f"Failed to log PyTorch model: {str(e)}",
                flavor="pytorch",
                details={"model_type": type(model).__name__, "artifact_name": name}
            ) from e

    def load_model(self, model_uri: str) -> Any:
        if not PYTORCH_AVAILABLE:
            raise ModelFlavorError(
                "PyTorch is not available. Please install it to load PyTorch models.",
                flavor="pytorch"
            )
        try:
            return mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            raise ModelFlavorError(
                f"Failed to load PyTorch model: {str(e)}",
                flavor="pytorch",
                details={"model_uri": model_uri}
            ) from e
    

class LightGBMFlavorHandler(FlavorHandler):
    """Handler for LightGBM models."""
    
    def can_handle(self, model: Any) -> bool:
        if not LIGHTGBM_AVAILABLE or lgb is None:
            return False
        return isinstance(model, (lgb.Booster, lgb.LGBMModel))

    def flavor(self) -> ModelFlavor:
        return ModelFlavor.LIGHTGBM

    def log_model(self, model: Any, name: str, **kwargs: Any) -> None:
        if not LIGHTGBM_AVAILABLE:
            raise ModelFlavorError(
                "LightGBM is not available. Please install it to log LightGBM models.",
                flavor="lightgbm"
            )
        try:
            mlflow.lightgbm.log_model(model, name, **kwargs)
        except Exception as e:
            raise ModelFlavorError(
                f"Failed to log LightGBM model: {str(e)}",
                flavor="lightgbm",
                details={"model_type": type(model).__name__, "artifact_name": name}
            ) from e

    def load_model(self, model_uri: str) -> Any:
        if not LIGHTGBM_AVAILABLE:
            raise ModelFlavorError(
                "LightGBM is not available. Please install it to load LightGBM models.",
                flavor="lightgbm"
            )
        try:
            return mlflow.lightgbm.load_model(model_uri)
        except Exception as e:
            raise ModelFlavorError(
                f"Failed to load LightGBM model: {str(e)}",
                flavor="lightgbm",
                details={"model_uri": model_uri}
            ) from e
    

class TransformersFlavorHandler(FlavorHandler):
    """Handler for Hugging Face Transformers models."""
    
    def can_handle(self, model: Any) -> bool:
        if not TRANSFORMERS_AVAILABLE:
            return False
        return isinstance(model, PreTrainedModel)

    def flavor(self) -> ModelFlavor:
        return ModelFlavor.TRANSFORMERS

    def log_model(self, model: Any, name: str, **kwargs: Any) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ModelFlavorError(
                "Transformers is not available. Please install it to log Transformers models.",
                flavor="transformers"
            )
        try:
            mlflow.transformers.log_model(model, name, **kwargs)
        except Exception as e:
            raise ModelFlavorError(
                f"Failed to log Transformers model: {str(e)}",
                flavor="transformers",
                details={"model_type": type(model).__name__, "artifact_name": name}
            ) from e

    def load_model(self, model_uri: str) -> Any:
        if not TRANSFORMERS_AVAILABLE:
            raise ModelFlavorError(
                "Transformers is not available. Please install it to load Transformers models.",
                flavor="transformers"
            )
        try:
            return mlflow.transformers.load_model(model_uri)
        except Exception as e:
            raise ModelFlavorError(
                f"Failed to load Transformers model: {str(e)}",
                flavor="transformers",
                details={"model_uri": model_uri}
            ) from e