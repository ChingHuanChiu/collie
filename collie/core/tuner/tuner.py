from collie.contracts.event import (
    Event, 
    EventHandler, 
    EventType
)
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.decorator import type_checker
from collie.core.models import (
    TunerArtifact,
    TunerPayload
)


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
        ) as run:
            tuner_event = self._handle(event)

            tuner_payload = self._tuner_payload(tuner_event)
            hyperparameters = tuner_payload.model_dump()

            self.log_dict(
                dictionary=hyperparameters, 
                artifact_path=TunerArtifact().hyperparameters
            )
            event.context.set(
                "hyperparameters_uri",
                self.artifact_path(run)
            )

            event_type = EventType.TUNING_DONE

            return Event(
                type=event_type,
                payload=tuner_payload,
                context=event.context
            )

    @type_checker((TunerPayload,) , 
        "TunerPayload must be of type TunerPayload."
    )    
    def _tuner_payload(self, event: Event) -> TunerPayload: 

        tuner_payload: TunerPayload = event.payload

        return tuner_payload
    
    def artifact_path(self, run) -> str:
        return f"{run.info.artifact_uri}/{TunerArtifact().hyperparameters}"