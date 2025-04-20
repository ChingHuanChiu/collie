from typing import (
    Dict, 
    Any,
    Optional,
    Union
)

from pydantic import Field, BaseModel

from collie._common.types import (
    EventType,
    TransformerPayload,
    TrainerPayload,
    TunerPayload,
    EvaluatorPayload
)
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
    

class Event(BaseModel):
    type: Optional[EventType] = None
    payload: Union[
        TransformerPayload, TrainerPayload, 
        TunerPayload, EvaluatorPayload
    ]
    context: PipelineContext = Field(default_factory=PipelineContext)


class _EventHandler:
    
    def handle(self, event: Event) -> Event:
        raise NotImplementedError()
    
    @type_checker((Event,), "The return type of *handle* method must be 'Event'.")
    def _handle(self, event: Event) -> Event:
        return self.handle(event)