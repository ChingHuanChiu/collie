from typing import (
    Union, 
    Optional,
    Dict, 
    Any
)
from abc import abstractmethod

from mlflow.tracking import MlflowClient

from collie._common.types import CollieComponents
from collie.contracts.mlflow import MLFlowComponentABC


class OrchestratorABC(MLFlowComponentABC):

    def __init__(
        self,
        components: Union[
            CollieComponents.TRAINER.value, 
            CollieComponents.TRANSFORMER.value,
            CollieComponents.TUNER.value,
            CollieComponents.EVALUATOR.value
        ],
        tracking_uri: Optional[str] = None,
        mlflow_tags: Optional[Dict[str, str]] = None,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:

        super().__init__()
        self.components = components
        self.mlflow_tags = mlflow_tags
        self.tracking_uri = tracking_uri
        self.description = description
        self.experiment_name = experiment_name
        
        if not self.tracking_uri:
            # TODO: use the better way
            self.tracking_uri = "./metadata/"
        self.mlflow_client = MlflowClient(tracking_uri=self.tracking_uri)

    @abstractmethod
    def run_pipeline(self) -> Any:
        raise NotImplementedError

    def run(self) -> None:

        self.set_tracking_uri(self.tracking_uri)
        self.set_experiment(self.experiment_name)
        experiment_id = self.get_exp_id(self.experiment_name)

        with self.start_run(
            tags=self.mlflow_tags, 
            run_name="Orchestrator", 
            description=self.description, 
            experiment_id=experiment_id
        ):

            self.run_pipeline()