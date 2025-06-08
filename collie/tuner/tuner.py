from collie.contracts.event import Event, EventHandler
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.types import (
    EventType,
    TunerPayload,
)
from collie._common.decorator import type_checker


class Tuner(EventHandler, MLFlowComponentABC):

    def __init__(self) -> None:
        super().__init__()
    
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

            tuner_payload = self._tuner_payload(tuner_event)
            event_type = EventType.TUNING_DONE
            # event.context.set("tuner_payload", tuner_payload)

            return Event(
                type=event_type,
                payload=tuner_payload,
                context=event.context
            )

    @type_checker((TunerPayload,) , 
        "TunerPayload must be of type TunerPayload."
    )    
    def _tuner_payload(self, event: Event) -> TunerPayload: 

        return event.payload