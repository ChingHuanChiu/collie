from collie.contracts.event import Event, EventHandler
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.types import (
    EventType,
    TransformerPayload,
)
from collie._common.decorator import type_checker


class Transformer(EventHandler, MLFlowComponentABC):

    def __init__(self) -> None:
        super().__init__()
    
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

            transformer_payload = self._transformer_payload(transformer_event)
            event_type = EventType.DATA_READY
            # event.context.set("transformer_payload", transformer_payload)

            return Event(
                type=event_type,
                payload=transformer_payload,
                context=event.context
            )
        
    @type_checker((TransformerPayload,) , 
        "TransformerPayload must be of type TransformerPayload."
    )    
    def _transformer_payload(self, event: Event) -> TransformerPayload: 

        return event.payload