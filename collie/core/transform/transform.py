from collie.contracts.event import (
    Event, 
    EventHandler, 
    EventType
)
from collie.contracts.mlflow import MLFlowComponentABC
from collie.core.models import (
    TransformerPayload,
    TransformerArtifactPath
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
        ) as run:
            transformer_event = self._handle(event)

            transformer_payload = self._transformer_payload(transformer_event)
            event_type = EventType.DATA_READY
            artifact_root = run.info.artifact_uri

            data = transformer_payload.model_dump()
            for data_type, data in data.items():

                if data is not None:
                    artifact_path = TransformerArtifactPath().model_dump()[data_type]
                    source = f"{artifact_root}/{data_type}.csv" 
                    event.context.set(
                        f"{data_type}_uri",
                        source
                    )
                    
                    
                    self.log_pd_data(
                        data=data, 
                        context=data_type,
                        source=source
                    )

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