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