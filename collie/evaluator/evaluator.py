from abc import abstractmethod
from typing import (
    Any, 
    Dict, 
    Union, 
    Literal, 
    List
)

from collie.abstract.mlflow import MLFlowComponentABC
from collie._common.mixin import OutputMixin
from collie._common.types import (
    EvaluatorDataModel,
    ComponentOutput
)


class Evaluator(MLFlowComponentABC, OutputMixin):
    
    def __init__(
        self, 
        registered_model_name: str, 
        model_uri: str
    ) -> None:
        
        super().__init__()
        self.registered_model_name = registered_model_name
        self.model_uri = model_uri

    @abstractmethod
    def evaluate(
        self, 
        outputs: ComponentOutput
    ) -> Any:
        
        raise NotImplementedError("Please implement the **evaluate** method.")
    
    def run(self) -> None:
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
            evaluation_data = self.evaluate(self.outputs)
            evaluation_results = EvaluatorDataModel(data=evaluation_data).data

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

            self.outputs: ComponentOutput = {"Evaluator": evaluation_results}
    
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