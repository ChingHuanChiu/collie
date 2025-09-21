from typing import List

from collie.contracts.event import (
    Event, 
    EventHandler, 
    EventType
)
from collie.contracts.mlflow import MLFlowComponentABC
from collie.core.models import (
    TrainerPayload,
    TrainerArtifactPath
)
from collie._common.decorator import type_checker


class Trainer(EventHandler, MLFlowComponentABC):

    def __init__(self) -> None:
        super().__init__()
    
    def run(self, event: Event) -> Event:
        """
        Run the trainer component.

        This method starts a new MLflow run, trains the model,
        logs metrics, and sets the outputs.
        """
        with self.start_run(
            run_name="Trainer", 
            tags={"component": "Trainer"},
            log_system_metrics=True, 
            nested=True
        ) as run:
            trainer_event = self._handle(event)

            trainer_payload = self._trainer_payload(trainer_event)
            event_type = EventType.TRAINING_DONE
            
            model = trainer_payload.model
            model_name = "model"
            self.log_model(
                model=model, 
                name=model_name
            )
            # artifacts = self.fetch_artifact_path(run)
            # model_uri = artifacts[0]
            model_uri = f"runs:/{run.info.run_id}/{model_name}"
            event.context.set("model_uri", model_uri)

            return Event(
                type=event_type,
                payload=trainer_payload,
                context=event.context
            )

    @type_checker((TrainerPayload,) , 
        "TrainerPayload must be of type TrainerPayload."
    )    
    def _trainer_payload(self, event: Event) -> TrainerPayload: 

        return event.payload
    
    def fetch_artifact_path(
        self, 
        run,
        # artifact_path: str
    ) -> List[str]:
        
        # Reference the mlflow.log_model method to list artifacts
        for artifact_path in ["model/data"]:
            # remove the "model/" prefix
            # TODO : Find the reason that the aritfiacts path are different between local and ui
            artifacts = [
                f.path[6:] for f in self.mlflow_client.list_artifacts(run.info.run_id, artifact_path)
            ]
        print(666666, artifacts)
        return artifacts