from typing import (
    Dict, 
    Any,
    Optional,
    TypeVar
)
from abc import abstractmethod, ABC
from pydantic import (
    Field, 
    BaseModel, 
    ConfigDict
)
from enum import Enum, auto

from collie._common.decorator import type_checker


class PipelineContext:
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = data or {}

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value

    def to_dict(self):
        return self.data

class EventType(Enum):
    INITIALIZE = auto()
    DATA_READY = auto()
    TRAINING_DONE = auto()
    TUNING_DONE = auto()
    EVALUATION_DONE = auto()
    PUSHER_DONE = auto()
    ERROR = auto()


P = TypeVar("P")
class Event(BaseModel):
    type: Optional[EventType] = None
    payload: P
    _context: PipelineContext = Field(default_factory=PipelineContext, alias="context")

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
    
    @property
    def context(self) -> PipelineContext:
        """
        Access the pipeline context (for framework internal use).
        
        Warning:
            Context should only be used for metadata (timestamps, versions, etc.).
            Use payload for passing data between components.
        
        Returns:
            PipelineContext instance
        """
        return self._context
    
    @context.setter
    def context(self, value: PipelineContext):
        """Set the context (for framework internal use)."""
        self._context = value


class EventHandler(ABC):
    
    @abstractmethod
    def handle(self, event: Event) -> Event:
        """
        Handle the incoming event and return a new event.
        
        This method must be implemented by all concrete event handlers.
        It should process the event payload and return a new event with
        the appropriate type and payload for the next component.
        
        Args:
            event (Event): The incoming event to process
            
        Returns:
            Event: A new event with processed payload
            
        Raises:
            NotImplementedError: If not implemented by concrete class
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the 'handle' method. "
            f"This method should process the incoming event and return a new event "
            f"with the appropriate payload for the next pipeline component."
        )
    
    @type_checker((Event,), "The return type of *handle* method must be 'Event'.")
    def _handle(self, event: Event) -> Event:
        """Internal wrapper for handle method with type checking."""
        return self.handle(event)