import pandas as pd
from typing import Dict

from collie.contracts.event import Event, EventHandler
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.types import (
    EventType,
    TransformerPayload,
    TransformerArtifactPath,
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

            data = transformer_payload.model_dump()
            for data_type, data in data.items():
                if data:
                    artifact_path = TransformerArtifactPath.model_dump()[data_type]
                    #TODO: remove the code after testing the log_text is working
                    # local_path = LocalArtifactDir.artifacts
                    self.log_data(
                        data=data, 
                        # local_path=f"{local_path}/Transformer/{data_type}.csv", 
                        artifact_path=artifact_path
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
    
    def log_data(
            self, 
            data: pd.DataFrame,
            # local_path: str,
            artifact_path: TransformerArtifactPath
        ) -> None:
        #TODO: remove the code after testing the log_text is working
        # data.to_csv(local_path, index=False)
        # self.log_artifact(
        #     local_path=local_path, 
        #     artifact_path=artifact_path
        # )
        self.log_text(
            text=data.to_csv(index=False),
            artifact_path=artifact_path
        )