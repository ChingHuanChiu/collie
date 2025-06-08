from typing import Union, Optional

from mlflow.tracking import MlflowClient

from collie._common.types import CollieComponents
from collie.contracts.mlflow import MLFlowComponentABC


class PipelineABC(MLFlowComponentABC):

    def __init__(self,
                tracking_uri: str,
                components: Union[
                    CollieComponents.TRAINER.value, 
                    CollieComponents.TRANSFORMER.value,
                    CollieComponents.TUNER.value,
                    CollieComponents.EVALUATOR.value
                ],
                experiment_name: Optional[str] = None,
                description: Optional[str] = None) -> None:

        super().__init__()
        self.components = components
        self.tracking_uri = tracking_uri
        self.description = description
        self.experiment_name = experiment_name

        self.mlflow_client = MlflowClient(tracking_uri=tracking_uri)

    def run(self) -> None:

        self.set_tracking_uri(self.tracking_uri)
        self.set_experiment(self.experiment_name)
        experiment_id = self.get_exp_id(self.experiment_name)

        with self.start_run(
            tags={"component": "LocalPipeline"}, 
            run_name="Pipeline", 
            description=self.description, 
            experiment_id=experiment_id
        ):

            for component in self.components:
                component.mlflow_client = self.mlflow_client
                _ = component.run()
            component.clear()