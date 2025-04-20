from abc import abstractmethod

from collie.contracts.event import Event, _EventHandler
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.types import (
    EventType,
    TrainerPayload,
)


class Trainer(_EventHandler, MLFlowComponentABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def handle(self, event: Event) -> Event:
        raise NotImplementedError("Please implement the **transform** method.")
    
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

            trainer_payload: TrainerPayload = trainer_event.payload
            event_type = EventType.TRAINING_DONE
            event.context.set("trainer_payload", trainer_payload)

            return Event(
                type=event_type,
                payload=trainer_payload,
                context=event.context
            )