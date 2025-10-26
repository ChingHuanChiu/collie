import tempfile
import threading
from abc import ABC, abstractmethod
from typing import (
    Any, 
    Optional, 
    Dict, 
    Literal, 
    List,
    Union,
    Generator,
    overload
)
from contextlib import contextmanager

import mlflow
import mlflow.data
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow import ActiveRun
from mlflow.exceptions import MlflowException

from collie._common.utils import get_logger
from collie._common.mlflow_model_io.model_io import MLflowModelIO
from collie._common.exceptions import (
    MLflowConfigurationError, 
    MLflowOperationError
)
from collie.core.enums.ml_models import (
    MLflowModelStage,
    ModelFlavor
)


logger = get_logger()


class MLflowConfig:
    """Singleton class to manage MLflow configuration."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
    ) -> None:
        
        if not self._initialized:
            self.tracking_uri = tracking_uri
            self.experiment_name = experiment_name
            self.mlflow_client = MlflowClient(tracking_uri=tracking_uri)
            self._initialized = True
    
    def configure(self) -> None:
        """Configure the singleton with MLflow settings."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)


class _MLflowModelManager:
    """Handles MLflow model management operations."""
    
    def __init__(
        self, 
        mlflow_client: MlflowClient
    ) -> None:
        
        self._mlflow_client = mlflow_client
        self._model_io = MLflowModelIO(mlflow_client)
    
    def log_model(self, model: Any, name: Optional[str] = None) -> None:
        
        """
        Logs a model with MLflow.

        Args:
            model (Any): The model to log.
            name (Optional[str], optional): The name to give the logged model. Defaults to None.

        Raises:
            MLflowOperationError: If logging the model fails.
        """
        try:
            self._model_io.log_model(model, name)
            logger.info(f"Logged model: {name or 'unnamed'}")
        except Exception as e:
            raise MLflowOperationError(f"Failed to log model '{name}': {e}") from e
    
    def load_model(
        self, 
        flavor: ModelFlavor,
        model_uri: Optional[str] = None
    ) -> Any:
        
        try:
            active_run = mlflow.active_run()
            if active_run is None:
                raise MLflowOperationError("No active run found")
            
            model = self._model_io.load_model(flavor, model_uri)
            return model
        except Exception as e:
            raise MLflowOperationError(f"Failed to load model '{model_uri}': {e}") from e
    
    def load_latest_model(
            self, 
            model_name: str, 
            stage: MLflowModelStage,
            flavor: ModelFlavor
        ) -> Any:
      
        """
        Loads the latest version of a model from a given stage.

        Args:
            model_name (str): The name of the model to load.
            stage (MLflowModelStage): The stage from which to load the model.
            flavor (ModelFlavor): The flavor of the model to load.

        Returns:
            Any: The loaded model if found, otherwise None.

        Raises:
            MLflowOperationError: If loading the model fails.
        """
        try:
            latest_versions = self._mlflow_client.get_latest_versions(model_name, stages=[stage])
            if not latest_versions:
                logger.warning(f"No model found in stage '{stage}' for model '{model_name}'")
                return None
            
            latest = latest_versions[0]  
            model_uri = latest.source 
        except Exception as e:
            logger.error(f"Failed to get latest model version for '{model_name}' in stage '{stage}': {e}")
            return None
            
        return self.load_model(flavor=flavor, model_uri=model_uri)

    def register_model(
        self, 
        model_name: str, 
        model_uri: str
    ) -> int:
        
        """
        Registers a model with MLflow.

        Args:
            model_name (str): The name to give the registered model.
            model_uri (str): The URI of the model to register.

        Returns:
            int: The version number of the registered model.

        Raises:
            MLflowOperationError: If registering the model fails.
        """
        
        try:
            registered_model = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model '{model_name}' version {registered_model.version}")
            return registered_model.version
        except MlflowException as e:
            raise MLflowOperationError(
                f"Failed to register model '{model_name}' with URI '{model_uri}': {e}"
            ) from e

    def transition_model_version(
        self,
        registered_model_name: str,
        version: str,
        desired_stage: str,
        archive_existing_versions_at_stage: bool = False,
    ) -> None:
        
        """
        Transitions a model version from one stage to another.

        Args:
            registered_model_name (str): The name of the registered model to transition.
            version (str): The version of the model to transition.
            desired_stage (str): The desired stage to transition the model to.
            archive_existing_versions_at_stage (bool, optional): Whether to archive existing versions at the target stage. Defaults to False.

        Raises:
            MLflowOperationError: If transitioning the model version fails.
        """
        try:
            self._mlflow_client.transition_model_version_stage(
                name=registered_model_name,
                version=version,
                stage=desired_stage,
                archive_existing_versions=archive_existing_versions_at_stage,
            )
            logger.info(f"Transitioned model '{registered_model_name}' v{version} to {desired_stage}")
        except MlflowException as e:
            raise MLflowOperationError(
                f"Failed to transition model '{registered_model_name}' v{version} to {desired_stage}: {e}"
            ) from e
        
    def get_latest_version(
        self,
        model_name: str,
        stages
    ) -> Optional[str]:
       
        """
        Retrieves the latest version number of a model in the specified stages.

        Args:
            model_name (str): The name of the model to retrieve the latest version for.
            stages (List[MLflowModelStage], optional): The stages in which to search for the latest version.

        Returns:
            Optional[str]: The latest version number of the model in the specified stages if found, otherwise None.

        Raises:
            MLflowOperationError: If retrieving the latest version fails.
        """
        try:
            latest_versions = self._mlflow_client.get_latest_versions(model_name, stages=stages)
            if not latest_versions:
                logger.warning(f"No versions found for model '{model_name}' in stages {stages}")
                return None

            latest_version = max(latest_versions, key=lambda v: int(v.version))
            return latest_version.version

        except MlflowException as e:
            raise MLflowOperationError(
                f"Failed to get latest version for model '{model_name}': {e}"
            ) from e


class MLFlowComponentABC(ABC):
    """
    Abstract base class for MLflow components with separated concerns.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mlflow_config = None
        self._model_manager = None
        self.mlflow = mlflow

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Please implement the **run** method.")
    
    @property
    def mlflow_config(self) -> MLflowConfig:
        
        if self._mlflow_config is None:
            raise MLflowConfigurationError("MLflow client not set") 
        return self._mlflow_config
    
    @mlflow_config.setter
    def mlflow_config(self, mlflow_config: MLflowConfig):
        if not isinstance(mlflow_config, MLflowConfig):
            raise MLflowConfigurationError("mlflow_config must be an instance of MLflowConfig")
        self._mlflow_config = mlflow_config

    @property
    def model_manager(self) -> Optional[_MLflowModelManager]:

        if self._model_manager is None:
            mlflow_client = self._mlflow_config.mlflow_client
            self._model_manager = _MLflowModelManager(mlflow_client)
        return self._model_manager

    def log_pd_data(
        self, 
        data: pd.DataFrame, 
        context: str, 
        source: str
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        
        try:
            ds = self.mlflow.data.from_pandas(data, source=source)
            self.mlflow.log_input(ds, context=context)

            with tempfile.NamedTemporaryFile(delete=True, suffix=".csv") as tmp:
                data.to_csv(tmp.name, index=False)
                self.mlflow.log_artifact(tmp.name)

            logger.debug(f"Logged pandas data for context: {context}")
        except Exception as e:
            raise MLflowOperationError(f"Failed to log pandas data: {e}") from e

    def log_model(
        self, 
        model: Any, 
        name: Optional[str] = None
    ) -> None:

        self.model_manager.log_model(model, name)
    
    def load_model(
        self, 
        name: Optional[str] = None
    ) -> Any:
       
        return self.model_manager.load_model(name)
    
    def load_latest_model(
            self, 
            model_name: str,
            flavor: ModelFlavor,
            stage: MLflowModelStage = MLflowModelStage.PRODUCTION,
        ) -> Any:
        
        return self.model_manager.load_latest_model(
            model_name, 
            stage,
            flavor
        ) 

    def register_model(
        self, 
        model_name: str, 
        model_uri: str
    ) -> int:
        
        return self.model_manager.register_model(model_name, model_uri)

    def transition_model_version(
        self, 
        registered_model_name: str, 
        version: str, 
        desired_stage: str, 
        archive_existing_versions_at_stage: bool = False
    ) -> None:
       
        self.model_manager.transition_model_version(
            registered_model_name, 
            version, 
            desired_stage, 
            archive_existing_versions_at_stage
        )
        
    def get_latest_version(
        self, 
        model_name: str, 
        stages: List[Literal["None", "Staging", "Production", "Archived"]]
    ) -> int:
        
        return self.model_manager.get_latest_version(model_name, stages)

    @overload
    def get_experiment(
        self, 
        return_id: Literal[True]
    ) -> Optional[str]: ...

    @overload
    def get_experiment(
        self, 
        return_id: Literal[False] = False
    ) -> Optional[mlflow.entities.Experiment]: ...
    
    def get_experiment(
        self,
        return_id: bool = False
    ) -> Optional[Union[mlflow.entities.Experiment, str]]:

        """
        Retrieves the MLflow experiment corresponding to the configured experiment name.

        Args:
            return_id (bool, optional): If True, returns the experiment ID instead of the experiment object. 
                                        Defaults to False.

        Returns:
            Optional[Union[mlflow.entities.Experiment, str]]: The experiment object or experiment ID if return_id is True, 
                                                                or None if the experiment does not exist.
    """
        experiment_name = self.mlflow_config.experiment_name
        if not experiment_name:
            return None
        try:
            experiment = self.mlflow.get_experiment_by_name(experiment_name)
            if return_id:
                return experiment.experiment_id
            else:
                return experiment
        except MlflowException as e:
            logger.error(f"Failed to get experiment '{experiment_name}': {e}")
            return None
        

    @contextmanager
    def start_run(
        self,
        tags: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
        nested: bool = False,
        log_system_metrics: Optional[bool] = None,
        description: Optional[str] = None,
    ) -> Generator[ActiveRun, None, None]:
        """
        Starts an MLflow run and returns the active run object.

        Args:
            tags (Optional[Dict[str, str]], optional): A dictionary of string key-value pairs to store as run tags. Defaults to None.
            run_name (Optional[str], optional): Name for the run. Defaults to None.
            nested (bool, optional): If True, nested runs are enabled. Defaults to False.
            log_system_metrics (Optional[bool], optional): If True, system metrics are logged. Defaults to None.
            description (Optional[str], optional): A string description for the run. Defaults to None.

        Yields:
            Generator[ActiveRun, None, None]: The active run object.
        Raises:
            MLflowOperationError: If the MLflow run cannot be started.
        """
        try:
            self.mlflow_config.configure()
            experiment_id = self.get_experiment(return_id=True)

            with self.mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                nested=nested,
                tags=tags,
                log_system_metrics=log_system_metrics,
                description=description,
            ) as active_run:
                logger.info(f"Started MLflow run: {active_run.info.run_id}")
                yield active_run

        except MlflowException as e:
            raise MLflowOperationError(f"Failed to start MLflow run: {e}") from e