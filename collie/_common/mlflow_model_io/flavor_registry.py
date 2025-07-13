from typing import List, Optional, Type

from collie._common.mlflow_model_io.base_flavor_handler import FlavorHandler
from collie._common.mlflow_model_io.model_flavors import (
    SklearnFlavorHandler,
    XGBoostFlavorHandler,
    PyTorchFlavorHandler,
    LightGBMFlavorHandler,
    TransformersFlavorHandler
)



class FlavorRegistry:
    def __init__(self):
        self._handlers: List[FlavorHandler] = [
            SklearnFlavorHandler(),
            XGBoostFlavorHandler(),
            PyTorchFlavorHandler(),
            LightGBMFlavorHandler(),
            TransformersFlavorHandler(),
        ]

    def find_handler_by_model(self, model) -> Optional[FlavorHandler]:
        for handler in self._handlers:
            if handler.can_handle(model):
                return handler
        return None

    def find_handler_by_flavor(self, flavor: str) -> Optional[FlavorHandler]:
        for handler in self._handlers:
            if handler.flavor() == flavor:
                return handler
        return None
