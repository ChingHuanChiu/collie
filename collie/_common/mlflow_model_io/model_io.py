from typing import Any, Optional

import mlflow

from mlflow.tracking import MlflowClient
from collie._common.mlflow_model_io.flavor_registry import FlavorRegistry


class MLflowModelIO:
    def __init__(self, mlflow_client: MlflowClient):
        self.registry = FlavorRegistry()
        self.client = mlflow_client

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        handler = self.registry.find_handler_by_model(model)
        if handler is None:
            raise ValueError(f"Unsupported model type: {type(model)}")

        handler.log_model(model, artifact_path, registered_model_name=registered_model_name, **kwargs)
        mlflow.log_param("model_flavor", handler.flavor())

    def load_model(
        self, 
        run_id: str, 
        artifact_path: str = "model"
    ) -> Any:
        
        run = self.client.get_run(run_id)
        flavor = run.data.params.get("model_flavor")

        if not flavor:
            raise ValueError(f"No model_flavor param found in run {run_id}")

        handler = self.registry.find_handler_by_flavor(flavor)
        if handler is None:
            raise ValueError(f"Unsupported model flavor: {flavor}")

        return handler.load_model(artifact_path)
