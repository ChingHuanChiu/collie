from collie.contracts.event import (
    Event, 
    EventHandler, 
    EventType
)
from collie.contracts.mlflow import MLFlowComponentABC
from collie.core.models import (
    TrainerPayload,
    TrainerArtifact
)
from collie._common.decorator import type_checker
from collie._common.exceptions import TrainerError 


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
            try:
                trainer_event = self._handle(event)

                trainer_payload = self._trainer_payload(trainer_event)
                event_type = EventType.TRAINING_DONE
                
                model = trainer_payload.model
                self.log_model(
                    model=model, 
                    name=TrainerArtifact().model
                )
                event.context.set("model_uri", self.artifact_uri(run))

                return Event(
                    type=event_type,
                    payload=trainer_payload,
                    context=event.context
                )
            except Exception as e:
                raise TrainerError(f"Trainer failed with error: {e}") from e

    @type_checker((TrainerPayload,) , 
        "TrainerPayload must be of type TrainerPayload."
    )    
    def _trainer_payload(self, event: Event) -> TrainerPayload: 

        return event.payload
    
    def artifact_uri(self, run) -> str:
        return f"runs:/{run.info.run_id}/{TrainerArtifact().model}"