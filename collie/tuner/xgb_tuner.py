from typing import Dict, Any

from hyperopt import fmin, tpe
import mlflow

from collie.tuner.tuner import Tuner


class XGBTuner(Tuner):

    def __init__(self,
                hyper_space: Dict[str, Any],
                max_evals: int) -> None:
        """
        Args:
            hyper_space (Dict[str, Any]): The hyperparameter search space.
            max_evals (int): The maximum number of hyperparameter evaluations to run.
        """
        super().__init__()
    
        self.space = hyper_space
        self.max_evals = max_evals

    def tune(self) -> Dict[str, Any]:
        """
        Run the hyperparameter tuner component.

        This method starts a new MLflow run, tunes the hyperparameters,
        logs metrics, and sets the outputs.

        Returns:
            Dict[str, Any]: The best hyperparameters for the model.
        """
        mlflow.xgboost.autolog(
            importance_types=["gain", "cover", "weight"],
            log_input_examples=False,
            log_model_signatures=True,
            log_models=True,
            log_datasets=False,
            disable=False,
            exclusive=False,
            disable_for_unsupported_versions=False,
            silent=False,
            registered_model_name=None,
            model_format="xgb",
            extra_tags=None
        )
        best_params = fmin(fn=self._objective, space=self.space, algo=tpe.suggest, max_evals=self.max_evals)

        return best_params
    




    from abc import abstractmethod

from collie.contracts.event import Event, EventHandler
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.decorator import type_checker
from collie._common.types import (
    EventType,
    TunerPayload,
)


# class Tuner(EventHandler, MLFlowComponentABC):

#     def __init__(self) -> None:
#         super().__init__()
        
#     abstractmethod
#     def handle(self, event: Event) -> Event:
#         raise NotImplementedError("Please implement the **transform** method.")
    
#     @abstractmethod
#     def objective(
#         self, 
#         outputs: ComponentOutput,
#         params: Dict[str, Any]
#     ) -> Dict[str, Any]:
#         """
#         Evaluate the objective function for hyperparameter tuning.

#         Args:
#             outputs (ComponentOutput): The output from the Transformer component.
#             params (Dict[str, Any]): A dictionary of hyperparameters to be used for training the model.

#         Returns:
#             Dict[str, Any]: A dictionary containing the negative average validation score as 'loss', 
#                             the hyperparameters used as 'params', and the status of the evaluation.
#         """
#         raise NotImplementedError("Please implement the **objective** method.")

#     @dict_key_checker(["loss", "params", "status"])
#     def _objective(self, param):

#         return self.objective(self.outputs, param)
    
#     def run(self) -> None:
#         """
#         Run the hyperparameter tuner component.

#         This method starts a new MLflow run, tunes the hyperparameters,
#         logs metrics, and sets the outputs.
#         """

#         with self.start_run(
#             tags={"component": "Tuner"},
#             run_name="Tuner",
#             log_system_metrics=True,
#             nested=True,
#         ):
#             hyperparameters = self.tune()

#             self.outputs: ComponentOutput = {
#                 "Tuner": hyperparameters,
#             }