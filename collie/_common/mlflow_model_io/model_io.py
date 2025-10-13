from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient
from collie._common.mlflow_model_io.flavor_registry import FlavorRegistry


class MLflowModelIO:
    def __init__(
        self, 
        mlflow_client: MlflowClient
    ) -> None:
        """
        Initializes an MLflowModelIO instance.

        Args:
            mlflow_client (MlflowClient): The MLflowClient instance to use for logging models.

        """
        self.registry = FlavorRegistry()
        self.client = mlflow_client

    def log_model(
        self,
        model: Any,
        name: str,
        registered_model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Logs a model with MLflow.

        Args:
            model (Any): The model to log.
            name (str): The name to give the logged model.
            registered_model_name (Optional[str], optional): The name to give the registered model. Defaults to None.
            **kwargs (Any): Additional keyword arguments to pass to the flavor handler's log_model method.

        Raises:
            ValueError: If the model type is not supported by any flavor handler.

        """
        handler = self.registry.find_handler_by_model(model)
        if handler is None:
            raise ValueError(f"Unsupported model type: {type(model)}")

        handler.log_model(
            model, 
            name, 
            registered_model_name=registered_model_name, 
            **kwargs
        )
        mlflow.log_param("model_flavor", handler.flavor())

    def load_model(
        self, 
        flavor: str,
        model_uri: str,
    ) -> Any:

        handler = self.registry.find_handler_by_flavor(flavor)
        if handler is None:
            raise ValueError(f"Unsupported model flavor: {flavor}")

        return handler.load_model(model_uri)