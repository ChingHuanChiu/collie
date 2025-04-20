from abc import abstractmethod

from collie.contracts.event import Event, _EventHandler
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.types import (
    EventType,
    TunerPayload,
)


class Tuner(_EventHandler, MLFlowComponentABC):

    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def handle(self, event: Event) -> Event:
        raise NotImplementedError("Please implement the **transform** method.")
    
    def run(self, event: Event) -> None:
        """
        Run the hyperparameter tuner component.

        This method starts a new MLflow run, tunes the hyperparameters,
        logs metrics, and sets the outputs.
        """

        with self.start_run(
            tags={"component": "Tuner"},
            run_name="Tuner",
            log_system_metrics=True,
            nested=True,
        ):
            tuner_event = self._handle(event)

            tuner_payload: TunerPayload = tuner_event.payload
            event_type = EventType.TUNING_DONE
            event.context.set("tuner_payload", tuner_payload)

            return Event(
                type=event_type,
                payload=tuner_payload,
                context=event.context
            )