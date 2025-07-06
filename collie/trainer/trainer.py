from collie.contracts.event import Event, EventHandler
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.types import (
    EventType,
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
        ):
            trainer_event = self._handle(event)

            trainer_payload = self._trainer_payload(trainer_event)
            event_type = EventType.TRAINING_DONE
            
            model = trainer_payload.model
            self.log_model(
                model=model, 
                artifact_path=TrainerArtifactPath.model
            )
            # event.context.set("trainer_payload", trainer_payload)

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