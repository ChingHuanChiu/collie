from typing import (
    Dict, 
    Any,
    List,
    Optional
)

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
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        
        """
        Initializes the Evaluator.

        Args:
            description (Optional[str], optional): Description for the MLflow run. Defaults to None.
            tags (Optional[Dict[str, str]], optional): Tags to associate with the MLflow run. Defaults to None.

        """
        super().__init__()
        self._registered_model_name = None
        self.description = description
        self.tags = tags or {"component": "Evaluator"}

        self.model_uri = None
        self.metrics = None
    
    @property
    def registered_model_name(self) -> str:
        if not self._registered_model_name:
            raise EvaluatorError("Registered model name is not set.")
        return self._registered_model_name
    
    @registered_model_name.setter
    def registered_model_name(self, name: str) -> None:
        self._registered_model_name = name
        
    def run(self, event: Event) -> Event:
       
        with self.start_run(
            tags=self.tags,
            run_name="Evaluator",
            log_system_metrics=True,
            description=self.description,
            nested=True,
        ) as run:
            
            try:
                evaluator_event = self._handle(event)
                payload = self._get_evaluator_payload(evaluator_event)

                self.metrics: List[Dict[str, Any]] = payload.metrics
                self.model_uri = event.context.get('model_uri')
        
                self._log_metrics()
                self._log_summary(payload)

                event.context.set("evaluator_report_uri", self.artifact_uri(run))

                if self.experiment_is_better(payload):
                    event.context.set("pass_evaluation", True)
                    self.mlflow.log_param("evaluation_result", "passed")
                else:
                    event.context.set("pass_evaluation", False)
                    self.mlflow.log_param("evaluation_result", "failed")
               
            except Exception as e:
                raise EvaluatorError(f"Evaluator failed: {e}") from e

            return Event(
                type=EventType.EVALUATION_DONE,
                payload=payload,
                context=event.context,
            )

    @staticmethod
    def experiment_is_better(
        payload: EvaluatorPayload
    ) -> bool:
        """
        Decide if experiment model is better based on score direction.
        """
        return payload.is_better_than_production
    
    def _flatten_metrics(
        self,
        metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Flatten list of metric dictionaries into a single dictionary.
        Useful for logging and analysis.
        """
        flattened = {}
        for metric_dict in metrics:
            flattened.update(metric_dict)
        return flattened

    def _log_summary(
        self, 
        payload: EvaluatorPayload
    ) -> None:
        """
        Generate summary statistics from metrics.
        """
        flattened = self._flatten_metrics(payload.metrics)
        
        summary = {
            "total_metrics": len(flattened),
            "is_better": payload.is_better_than_production,
            "metrics": flattened
        }
        self.mlflow.log_dict(
            dictionary=summary, 
            artifact_file=f"{EvaluatorArtifact().report}"
        )
    
    def _log_metrics(self) -> None:
        """
        Log individual metrics to MLflow.
        """
        for metric_dict in self.metrics:
            for metric_name, metric_value in metric_dict.items():
                self.mlflow.log_metric(metric_name, metric_value)
    
    @type_checker((EvaluatorPayload,), "EvaluatorPayload must be of type EvaluatorPayload.")
    def _get_evaluator_payload(self, event: Event) -> EvaluatorPayload: 
        return event.payload
    
    def artifact_uri(self, run) -> str:
        return f"{run.info.artifact_uri}/{EvaluatorArtifact().report}"