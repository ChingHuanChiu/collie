from abc import ABCMeta, abstractmethod
from typing import (
    Any, 
    Optional, 
    Dict, 
    Literal, 
    List,
    Union,
    Generator
)
from contextlib import contextmanager

import numpy as np
import mlflow
import PIL
from mlflow.tracking import MlflowClient
from mlflow import ActiveRun
import xgboost as xgb
import sklearn.base
import torch.nn as nn

from collie._common.mlflow_model_io.model_io import MLflowModelIO


class MLFlowComponentABC(metaclass=ABCMeta):

    _mlflow_client = None

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Please implement the **run** method.")
    
    @property
    def mlflow_client(self) -> MlflowClient:

        """
        Gets the MLflowClient instance for this component.

        Raises:
            ValueError: If the MLflowClient instance is not set.
        """
        if self._mlflow_client is None:
            raise ValueError("Please set the MLflowClient instance for this component.")
        return self._mlflow_client
    
    @mlflow_client.setter
    def mlflow_client(self, mlflow_client: MlflowClient) -> None:

        """
        Sets the MLflowClient instance for this component.

        If the MLflowClient is not provided, it will be set to None.
        """
        if self._mlflow_client is None:
            self._mlflow_client = mlflow_client
    
    def set_tracking_uri(self, tracking_uri: str) -> None:

        mlflow.set_tracking_uri(tracking_uri)

    def set_experiment(self, experiment_name: str) -> None:

        mlflow.set_experiment(experiment_name)
    
    def log_artifact(
        self, 
        local_path: str, 
        artifact_path: Optional[str] = None
    ) -> None:
        """
        Logs a local file or directory as an artifact of the currently active run.

        Args:
            local_path: Path to the file or directory to write.
            artifact_path: If provided, the directory in ``artifact_uri`` to write to.
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_metric(
        self, 
        key: str, 
        value: float, 
        step: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Logs a metric (e.g. model evaluation metric) with the given key and value.

        Args:
            key: Metric name. This string may only contain alphanumerics, underscores
                (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                All backend stores support keys up to length 250, but some may
                support larger keys.
            value: Metric value. Note that some special values such
                as +/- Infinity may be replaced by other values depending on the store. For
                example, the SQLAlchemy store replaces +/- Inf with max / min float values.
                All built-in backend stores support values up to length 5000, but some
                may support larger values.
            step: Integer training step (iteration) at which was the metric calculated.
                Defaults to 0.
            **kwargs: Additional keyword arguments to pass to mlflow.log_metric
        """
        mlflow.log_metric(
            key=key, 
            value=value, 
            step=step,
            **kwargs
        )

    def log_param(
        self, key: str, 
        value: str, 
        **kwargs
    ) -> None:
        """
        Logs a parameter as an artifact with the given keyword arguments.

        Args:
            key: Parameter name. This string may only contain alphanumerics, underscores
                (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                All backend stores support keys up to length 250, but some may
                support larger keys.
            value: Parameter value, but will be string-ified if not.
                All built-in backend stores support values up to length 6000, but some
                may support larger values.
            **kwargs: Additional keyword arguments to pass to mlflow.log_param
        """
        mlflow.log_param(key=key, value=value, **kwargs)

    def log_image(
        self, 
        image: Union["np.ndarray", "PIL.Image.Image", "mlflow.Image"],
        artifact_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Logs an image as an artifact with the given artifact path and keyword arguments.

        Args:
            image: The image to log. This can be either a numpy array, PIL Image, or an
                mlflow.Image.
            artifact_path: The artifact path to log the image to.
            **kwargs: Additional keyword arguments to pass to mlflow.log_image
        """
        mlflow.log_image(image, artifact_path, **kwargs)

    def log_text(
        self, 
        text: str, 
        artifact_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:

        """
        Logs text as an artifact with the given artifact path and keyword arguments.

        Args:
            text: The text to log.
            artifact_path: The artifact path to log the text to.
            **kwargs: Additional keyword arguments to pass to mlflow.log_text
        """
        mlflow.log_text(text, artifact_path, **kwargs)

    def load_text(
        self, 
        artifact_path: str
    ) -> str:
        return mlflow.artifacts.load_text(artifact_path)

    def log_dict(
        self,
        dictionary: Dict[str, Any], 
        artifact_path: Optional[str] = None,
        **kwargs: Any
    ) -> None: 

        """
        Logs a dictionary as an artifact with the given artifact path and keyword arguments.

        Args:
            dictionary: The dictionary to log.
            artifact_path: The artifact path to log the dictionary to.
            **kwargs: Additional keyword arguments to pass to mlflow.log_dict
        """
        mlflow.log_dict(dictionary, artifact_path, **kwargs)

    def load_dict(
        self, 
        artifact_path: str
    ) -> Dict[str, Any]:
        return mlflow.artifacts.load_dict(artifact_path)

    def log_model(
        self, 
        model: Any, 
        artifact_path: str, 
    ) -> None:
       
        model_io = MLflowModelIO()

        model_io.log_model(model, artifact_path)
    
    def load_model(
        self, 
        artifact_path: str
    ) -> Any:
        
        model_io = MLflowModelIO()
        run_id = mlflow.active_run().info.run_id
        return model_io.load_model(run_id, artifact_path)

    def transition_model_version(
        self,
        registered_model_name: str,
        version: str,
        desired_stage: str,
        archive_existing_versions_at_stage: bool = False,
    ) -> None:
        """
        Transition a model version to a specified stage.

        Args:
            registered_model_name (str): The name of the registered model.
            version (int): The version number of the model to transition.
            desired_stage (str): The desired stage for the model version.
            archive_existing_versions_at_stage (bool, optional):
                If True, all existing versions in the specified stage will be
                archived. Defaults to False.
        """
        self.mlflow_client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage=desired_stage,
            archive_existing_versions=archive_existing_versions_at_stage,
        )
    
    def register_model(self, model_name: str, model_uri: str) -> int:
        """
        Register a model with MLflow.

        Args:
            model_name (str): The name to give the registered model.
            model_uri (str): The URI of the model to register.

        Returns:
            int: The version number of the newly registered model.

        Raises:
            RuntimeError: If the registration fails.
        """
        try:
            registered_model = mlflow.register_model(model_uri, model_name)
            return registered_model.version
        except Exception as e:
            raise RuntimeError(
                f"Failed to register model '{model_name}' with URI '{model_uri}': {e}"
            ) from e
    
    @contextmanager
    def start_run(
        self,
        tags: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
        nested: bool = False,
        log_system_metrics: Optional[bool] = None,
        experiment_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Generator[ActiveRun, None, None]:
        """
        Context manager to start a new MLflow run.

        Args:
            tags (dict[str, str], optional): Additional metadata to record with the run.
            run_name (str, optional): Name of the run.
            nested (bool, optional): If True, the run is nested within the current active run.
            log_system_metrics (bool, optional): If True, MLflow will log system metrics
                (e.g. git commit, cwd, etc.) to the run.
            experiment_id (str, optional): The experiment ID under which to launch the run.
            description (str, optional): A description of the run.

        Yields:
            The active run object.
        """
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            nested=nested,
            tags=tags,
            log_system_metrics=log_system_metrics,
            description=description,
        ) as active_run:
            yield active_run
    
    def get_exp_id(self, experiment_name: str) -> str:
        """
        Get the MLflow experiment ID corresponding to the given experiment name.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            str: The experiment ID.

        Notes:
            If the experiment does not exist, it will be created.
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)
        
    def get_latest_version(
        self,
        model_name: str,
        stages: List[Literal["None", "Staging", "Production", "Archived"]],
    ) -> int:
        """
        Get the latest version of the model from the specified stages.

        Args:
            model_name: The name of the model.
            stages: The stages to query for the latest model version.

        Returns:
            The version of the latest model.
        """
        latest_versions = self.mlflow_client.get_latest_versions(
            model_name, stages=stages
        )
        if not latest_versions:
            return 0
        return latest_versions[0].version
