from abc import abstractmethod

from collie.contracts.event import Event, _EventHandler
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.types import (
    EventType,
    TransformerPayload,
)


class Transformer(_EventHandler, MLFlowComponentABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def handle(self, event: Event) -> Event:
        raise NotImplementedError("Please implement the **transform** method.")
    
    def run(self, event: Event) -> Event:
        """
        Run the transformer component.

        This method starts a new MLflow run, transforms the input data,
        logs metrics, and sets the outputs.
        """
        with self.start_run(
            tags={"component": "Transformer"},
            run_name="Transformer",
            log_system_metrics=True,
            nested=True,
        ):
            transformer_event = self._handle(event)

            transformer_payload: TransformerPayload = transformer_event.payload
            event_type = EventType.DATA_READY
            event.context.set("transformer_payload", transformer_payload)

            return Event(
                type=event_type,
                payload=transformer_payload,
                context=event.context
            )