from typing import Union, Optional

from mlflow.tracking import MlflowClient

from collie._common.enums import CollieComponents
from collie.abstract.mlflow import MLFlowComponentABC


class LocalPipeline(MLFlowComponentABC):
    """
    Defines a pipeline for executing a series of components 
    (e.g., trainer, transformer, tuner, evaluator) in a local environment.
    """

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

        """
        Args:
            tracking_uri (str): The MLflow tracking URI.
            components (Union[CollieComponents.TRAINER.value, 
                            CollieComponents.TRANSFORMER.value, 
                            CollieComponents.TUNER.value, 
                            CollieComponents.EVALUATOR.value]): 
                The components to be executed in the pipeline.
            experiment_name (Optional[str]): The name of the MLflow experiment. Defaults to None.
            description (Optional[str]): A description for the MLflow run. Defaults to None.
        """

        super().__init__()
        self.components = components
        self.tracking_uri = tracking_uri
        self.description = description
        self.experiment_name = experiment_name

        self.mlflow_client = MlflowClient(tracking_uri=tracking_uri)

    def run(self) -> None:
        """
        Run all components in the pipeline.

        This function will:
        1. Set the current MLflow tracking URI to the given tracking URI.
        2. Set the current MLflow experiment to the given experiment name.
        3. Get the ID of the MLflow experiment.
        4. Start a new run under the experiment with the given description.
        5. Run each component in the pipeline and log its metrics.
        6. Clear the MLflow client from each component.
        """
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