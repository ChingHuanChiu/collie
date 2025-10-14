from typing import (
    Optional,
    Dict, 
    Any
)
from abc import abstractmethod

from collie.contracts.event import Event, EventType
from collie.core.enums.components import CollieComponentType
from collie.contracts.mlflow import (
    MLFlowComponentABC,
    MLflowConfig
)
from collie._common.exceptions import (
    OrchestratorError,
    TrainerError,
    TunerError,
    EvaluatorError,
    PusherError,
    TransformerError,
)


class OrchestratorBase(MLFlowComponentABC):

    def __init__(
        self,
        components: CollieComponentType,
        tracking_uri: Optional[str] = None,
        mlflow_tags: Optional[Dict[str, str]] = None,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:

        super().__init__()
        self.components = components
        self.mlflow_tags = mlflow_tags
        self.tracking_uri = tracking_uri
        self.description = description
        self.experiment_name = experiment_name

    @abstractmethod
    def orchestrate_pipeline(self) -> Any:
        raise NotImplementedError

    def run(self) -> Any:
        
        self.mlflow_config = MLflowConfig(
            tracking_uri=self.tracking_uri,
            experiment_name=self.experiment_name,
        )
        try:
            with self.start_run(
                tags=self.mlflow_tags, 
                run_name="Orchestrator", 
                description=self.description, 
            ):
                return self.orchestrate_pipeline()
        except (
            TrainerError,
            TunerError,
            EvaluatorError,
            PusherError,
            TransformerError,
        ) as e:
            raise OrchestratorError(
                f"Component error in orchestration: {str(e)}"
            ) from e
        except Exception as e:
            raise OrchestratorError(
                f"Unexpected orchestration error: {str(e)}"
            ) from e

    def initialize_event(self) -> Event:
        """Initialize pipeline with an event."""
        return Event(
            type=EventType.INITIALIZE,
            payload=None
        )