from abc import ABC, abstractmethod
from typing import Any


class FlavorHandler(ABC):

    @abstractmethod
    def can_handle(self, model: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def flavor(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def log_model(self, model: Any, artifact_path: str, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_uri: str) -> Any:
        raise NotImplementedError
