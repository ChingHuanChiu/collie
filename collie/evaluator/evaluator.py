from typing import (
    List,
    Literal,
)
from abc import abstractmethod

from collie.contracts.event import Event, _EventHandler
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.types import (
    EventType,
    EvaluatorPayload
)
from collie._common.decorator import type_checker


class Evaluator(_EventHandler, MLFlowComponentABC):
    
    def __init__(
        self, 
        registered_model_name: str, 
        model_uri: str
    ) -> None:
        
        super().__init__()
        self.registered_model_name = registered_model_name
        self.model_uri = model_uri

    @abstractmethod
    def handle(self, event: Event) -> Event:
        pass
        
    def run(self, event: Event) -> Event:
        """
        Run the evaluator component.

        This method starts a new MLflow run, evaluates the model, logs metrics,
        transitions the model version to "Staging" if the experiment model score
        is better than the production model score, and sets the outputs.
        """

        with self.start_run(
            tags={"component": "Evaluator"},
            run_name="Evaluator",
            log_system_metrics=True,
            nested=True,
        ):
            
            evaluator_event = self._handle(event)

            evaluator_payload = self._evaluator_payload(evaluator_event)
            event_type = EventType.EVALUATION_DONE
            # event.context.set("evaluator_payload", evaluator_payload)
            evaluation_results = evaluator_payload.metrics


            self.register_model(model_name=self.registered_model_name, model_uri=self.model_uri)
            for metric_name, metric_value in evaluation_results.items():
                self.log_metric(metric_name, metric_value)

            experiment_model_score = evaluation_results.get("Experiment")
            production_model_score = evaluation_results.get("Production")

            if experiment_model_score >= production_model_score:
                version_to_transition = self._get_version_to_transition(stages=["Staging"])
                self.transition_model_version(
                    registered_model_name=self.registered_model_name,
                    version=version_to_transition,
                    desired_stage="Staging",
                    archive_existing_versions_at_stage=True,
                )
            else:
                version_to_transition = self._get_version_to_transition(stages=["None"])
                self.transition_model_version(
                    registered_model_name=self.registered_model_name,
                    version=version_to_transition,
                    desired_stage="None",
                    archive_existing_versions_at_stage=False,
                )

            return Event(
                type=event_type,
                payload=evaluator_payload,
                context=event.context,
            )
    
    def _get_version_to_transition(
        self,
        stages: List[Literal["None", "Staging", "Production", "Archived"]]
    ) -> str:
        """
        Get the version to transition for the given stages.

        Args:
            stages (List[Literal["None", "Staging", "Production", "Archived"]]): The stages to query for the latest model version.

        Returns:
            str: The version to transition.
        """
        versions = self.get_latest_version(self.registered_model_name, stages=stages)
        return str(versions + 1)
    
    @type_checker((EvaluatorPayload, ) , 
        "EvaluatorPayload must be of type EvaluatorPayload."
    )    
    def _evaluator_payload(self, event: Event) -> EvaluatorPayload: 

        return event.payload