from typing import Literal

from collie.contracts.event import (
    Event, 
    EventHandler, 
    EventType
)
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.decorator import type_checker
from collie.core.models import (
    EvaluatorArtifact,
    EvaluatorPayload
)
from collie._common.exceptions import EvaluatorError


class Evaluator(EventHandler, MLFlowComponentABC):
    
    def __init__(
        self, 
        registered_model_name: str, 
        model_uri: str
    ) -> None:
        super().__init__()
        self.registered_model_name = registered_model_name
        self.model_uri = model_uri

    def run(self, event: Event) -> Event:
        """
        Evaluate model, log metrics, and transition model version based on comparison.
        Args:
            event (Event): Event to handle

        Returns:
            Event: Event with type EventType.EVALUATION_DONE
        
        Raises:
            ValueError: If either 'Experiment' or 'Production' metrics are not provided.
        """
        with self.start_run(
            tags={"component": "Evaluator"},
            run_name="Evaluator",
            log_system_metrics=True,
            nested=True,
        ) as run:
            
            try:
                evaluator_event = self._handle(event)
                payload = self._get_evaluator_payload(evaluator_event)

                self.model_uri = event.context.get('model_uri')
                self.register_model(
                    model_name=self.registered_model_name, 
                    model_uri=self.model_uri
                )

                for metric_name, metric_value in payload.metrics.items():
                    self.log_metric(metric_name, metric_value)
                
                self.log_dict(
                    dictionary=payload.model_dump(), 
                    artifact_path=EvaluatorArtifact().report
                )
                event.context.set("evaluator_report_uri", self.artifact_uri(run))

                experiment_score = payload.metrics.get("Experiment")
                production_score = payload.metrics.get("Production")
                if experiment_score is None or production_score is None:
                    raise EvaluatorError("Both 'Experiment' and 'Production' metrics must be provided.")

                if self._is_experiment_better(
                    experiment_score, production_score, payload.greater_is_better
                ):
                    stage = "Staging"
                    archive = True
                else:
                    stage = "None"
                    archive = False

                version = self._next_model_version(stage)
                self.transition_model_version(
                    registered_model_name=self.registered_model_name,
                    version=version,
                    desired_stage=stage,
                    archive_existing_versions_at_stage=archive,
                )
            except Exception as e:
                raise EvaluatorError(f"Evaluator failed: {e}") from e

            return Event(
                type=EventType.EVALUATION_DONE,
                payload=payload,
                context=event.context,
            )

    @staticmethod
    def _is_experiment_better(
        experiment: float, 
        production: float, 
        greater_is_better: bool
    ) -> bool:
        """
        Decide if experiment model is better based on score direction.
        """
        return experiment >= production if greater_is_better else experiment <= production

    def _next_model_version(
        self,
        stage: Literal["None", "Staging", "Production", "Archived"]
    ) -> str:
        """
        Compute next version number for the given stage.
        """
        # Note: Must be used after register_model()
        # This assumes the registry starts with version=1
        versions = self.get_latest_version(self.registered_model_name, stages=[stage])
        return str(versions + 1)

    @type_checker((EvaluatorPayload,), "EvaluatorPayload must be of type EvaluatorPayload.")
    def _get_evaluator_payload(self, event: Event) -> EvaluatorPayload: 
        return event.payload
    
    def artifact_uri(self, run) -> str:
        return f"{run.info.artifact_uri}/{EvaluatorArtifact().report}"