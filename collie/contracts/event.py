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

#TODO: place the right file
class EventType(Enum):
    INITAILIZE = auto()
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
    context: PipelineContext = Field(default_factory=PipelineContext) #TODO: remove this field because the artifact is used

    model_config = ConfigDict(arbitrary_types_allowed=True)



class EventHandler(ABC):
    
    @abstractmethod
    def handle(self, event: Event) -> Event:
        raise NotImplementedError()
    
    @type_checker((Event,), "The return type of *handle* method must be 'Event'.")
    def _handle(self, event: Event) -> Event:
        return self.handle(event)