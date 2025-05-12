from typing import Dict, Any
from abc import abstractmethod

import mlflow
import xgboost as xgb

from collie.contracts.event import Event
from collie.trainer.trainer import Trainer
from collie._common.types import TrainerPayload, EventType
from collie._common.decorator import type_checker


class XGBTrainer(Trainer):

    def __init__(self) -> None:
        
        super().__init__()

    def run(self, event: Event) -> Event:
        """
        Train the XGBoost model.

        This method is a wrapper around _fit and is responsible for
        starting the MLflow run, training the model with the hyperparameters
        specified in the space, and logging metrics.

        Returns:
            Dict[str, Any]: A dictionary with a single key value pair where
                the key is "model" and the value is the trained XGBoost model.
        """
        with self.start_run(
            run_name="XGBTrainer", 
            tags={"component": "XGBTrainer"},
            log_system_metrics=True, 
            nested=True
        ):
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
            trainer_event = self._handle(event)
            trainer_payload = self._trainer_payload(trainer_event)
            event_type = EventType.TRAINING_DONE
            self._check_model_type(trainer_payload=trainer_payload)

            return Event(
                type=event_type,
                payload=trainer_payload,
                context=event.context
            )
    
    @type_checker((xgb.XGBClassifier, 
                 xgb.XGBModel, 
                 xgb.XGBRanker, 
                 xgb.XGBRegressor, 
                 xgb.XGBRFClassifier, 
                 xgb.XGBRFRegressor),
                 "Model must be one of the following types: "
                 "XGBClassifier, XGBModel, XGBRanker, "
                 "XGBRegressor, XGBRFClassifier, or XGBRFRegressor.")
    def _check_model_type(self, trainer_payload: TrainerPayload):
        model = trainer_payload.model
        return model