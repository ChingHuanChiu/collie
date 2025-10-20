from typing import List, Optional

from collie._common.mlflow_model_io.base_flavor_handler import FlavorHandler
from collie._common.mlflow_model_io.model_flavors import (
    SklearnFlavorHandler,
    XGBoostFlavorHandler, 
    PyTorchFlavorHandler,
    LightGBMFlavorHandler,
    TransformersFlavorHandler,
    SKLEARN_AVAILABLE,
    XGBOOST_AVAILABLE,
    PYTORCH_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    TRANSFORMERS_AVAILABLE
)
from collie._common.exceptions import ModelFlavorError


class FlavorRegistry:
    """Registry for model flavor handlers with conditional loading."""
    
    def __init__(self):
        self._handlers: List[FlavorHandler] = []
        
        # Only register handlers for available frameworks
        if SKLEARN_AVAILABLE:
            self._handlers.append(SklearnFlavorHandler())
        if XGBOOST_AVAILABLE:
            self._handlers.append(XGBoostFlavorHandler())
        if PYTORCH_AVAILABLE:
            self._handlers.append(PyTorchFlavorHandler())
        if LIGHTGBM_AVAILABLE:
            self._handlers.append(LightGBMFlavorHandler())
        if TRANSFORMERS_AVAILABLE:
            self._handlers.append(TransformersFlavorHandler())
            
        if not self._handlers:
            raise ModelFlavorError(
                "No model flavor handlers available. Please install at least one supported ML framework."
            )

    def find_handler_by_model(self, model) -> Optional[FlavorHandler]:
        """Find a handler that can handle the given model."""
        for handler in self._handlers:
            if handler.can_handle(model):
                return handler
        return None

    def find_handler_by_flavor(self, flavor: str) -> Optional[FlavorHandler]:
        """Find a handler by flavor name."""
        for handler in self._handlers:
            if handler.flavor() == flavor:
                return handler
        return None
    
    def get_available_flavors(self) -> List[str]:
        """Get list of available model flavors."""
        return [handler.flavor().value for handler in self._handlers]
    
    def get_handler_info(self) -> dict:
        """Get information about registered handlers."""
        return {
            "total_handlers": len(self._handlers),
            "available_flavors": self.get_available_flavors(),
            "framework_status": {
                "sklearn": SKLEARN_AVAILABLE,
                "xgboost": XGBOOST_AVAILABLE,
                "pytorch": PYTORCH_AVAILABLE,
                "lightgbm": LIGHTGBM_AVAILABLE,
                "transformers": TRANSFORMERS_AVAILABLE
            }
        }