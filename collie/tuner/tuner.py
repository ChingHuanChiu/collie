from typing import Any, Dict
from abc import abstractmethod

from collie.abstract.mlflow import MLFlowComponentABC
from collie._common.mixin import OutputMixin
from collie._common.types import ComponentOutput
from collie._common.decorator import dict_key_checker


class Tuner(MLFlowComponentABC, OutputMixin):

    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def tune(self) -> Dict[str, Any]:
        raise NotImplementedError("Please implement the *tunel* method.")
    
    @abstractmethod
    def objective(
        self, 
        outputs: ComponentOutput,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the objective function for hyperparameter tuning.

        Args:
            outputs (ComponentOutput): The output from the Transformer component.
            params (Dict[str, Any]): A dictionary of hyperparameters to be used for training the model.

        Returns:
            Dict[str, Any]: A dictionary containing the negative average validation score as 'loss', 
                            the hyperparameters used as 'params', and the status of the evaluation.
        """
        raise NotImplementedError("Please implement the **objective** method.")

    @dict_key_checker(["loss", "params", "status"])
    def _objective(self, param):

        return self.objective(self.outputs, param)
    
    def run(self) -> None:
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
            hyperparameters = self.tune()

            self.outputs: ComponentOutput = {
                "Tuner": hyperparameters,
            }