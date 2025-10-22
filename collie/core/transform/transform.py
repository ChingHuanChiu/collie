from typing import Optional
from collie.contracts.event import (
    Event, 
    EventHandler, 
    EventType
)
from collie.contracts.mlflow import MLFlowComponentABC
from collie.core.models import (
    TransformerPayload,
    TransformerArtifact
)
from collie._common.decorator import type_checker
from collie._common.exceptions import TransformerError


class Transformer(EventHandler, MLFlowComponentABC):

    def __init__(
        self,
        description: Optional[str] = None,
        tags: Optional[dict] = None
    ) -> None:
        """
        Initializes the Transformer component.

        Args:
            description (Optional[str], optional): Description for the MLflow run. Defaults to None.
            tags (Optional[dict], optional): Tags to associate with the MLflow run. Defaults to None.
        """
        super().__init__()
        self.description = description
        self.tags = tags or {"component": "Transformer"}
    
    def run(self, event: Event) -> Event:
        """
        Run the transformer component.

        This method starts a new MLflow run, transforms the input data,
        logs metrics, and sets the outputs.
        """
        with self.start_run(
            tags=self.tags,
            run_name="Transformer",
            log_system_metrics=True,
            nested=True,
            description=self.description
        ) as run:
            try:
                transformer_event = self._handle(event)

                transformer_payload = self._transformer_payload(transformer_event)
                event_type = EventType.DATA_READY

                data = transformer_payload.model_dump()
                for data_type, data in data.items():

                    if data is not None:
                        source = self.artifact_uri(run, data_type)
                        event._context.set(
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
                    _context=event._context
                )

            except Exception as e:
                raise TransformerError(f"Transformer failed with error: {e}")
        
    @type_checker((TransformerPayload,) , 
        "TransformerPayload must be of type TransformerPayload."
    )    
    def _transformer_payload(self, event: Event) -> TransformerPayload: 

        return event.payload
    
    def artifact_uri(self, run, data_type) -> str:
         return f"{run.info.artifact_uri}/{TransformerArtifact().model_dump()[data_type]}" 