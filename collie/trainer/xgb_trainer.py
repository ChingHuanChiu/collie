from typing import Dict, Any
from abc import abstractmethod

import mlflow
import xgboost as xgb

from collie.trainer.trainer import Trainer
from collie._common.decorator import type_checker
from collie._common.types import ComponentOutput


class XGBTrainer(Trainer):

    def __init__(self) -> None:
        
        super().__init__()

    @abstractmethod
    def fit_model(self, outputs: ComponentOutput) -> xgb:
        raise NotImplementedError("Please implement the *fit_model* method.")

    def train(self) -> xgb:
        """
        Train the XGBoost model.

        This method is a wrapper around _fit and is responsible for
        starting the MLflow run, training the model with the hyperparameters
        specified in the space, and logging metrics.

        Returns:
            Dict[str, Any]: A dictionary with a single key value pair where
                the key is "model" and the value is the trained XGBoost model.
        """
        mlflow.xgboost.autolog(
            importance_types=["gain", "cover", "weight"],
            log_input_examples=False,
            log_model_signatures=True,
            log_models=True,
            log_datasets=True,
            disable=False,
            exclusive=False,
            disable_for_unsupported_versions=False,
            silent=False,
            registered_model_name=None,
            model_format="xgb",
            extra_tags=None
        )
        
        model = self._fit()

        return {"model": model}
    
    @type_checker((xgb.XGBClassifier, 
                 xgb.XGBModel, 
                 xgb.XGBRanker, 
                 xgb.XGBRegressor, 
                 xgb.XGBRFClassifier, 
                 xgb.XGBRFRegressor),
                 "Model must be one of the following types: "
                 "XGBClassifier, XGBModel, XGBRanker, "
                 "XGBRegressor, XGBRFClassifier, or XGBRFRegressor.")
    def _fit(self):

        """
        A wrapper around the abstract *fit_model* method.

        This method starts a new MLflow run and logs the model.

        Returns:
            xgb.XGBClassifier|xgb.XGBModel|xgb.XGBRanker|xgb.XGBRegressor|xgb.XGBRFClassifier|xgb.XGBRFRegressor:
                The trained XGBoost model.

        Raises:
            TypeError: If the model returned by *fit_model* is not one of the
                following types: XGBClassifier, XGBModel, XGBRanker,
                XGBRegressor, XGBRFClassifier, or XGBRFRegressor.
        """
        model = self.fit_model(self.outputs)

        return model