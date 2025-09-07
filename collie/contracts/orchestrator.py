from typing import (
    Optional,
    Dict, 
    Any
)
from abc import abstractmethod

from mlflow.tracking import MlflowClient

from collie.contracts.event import Event
from collie._common.types import CollieComponentType, CollieComponents
from collie.contracts.mlflow import MLFlowComponentABC


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
        #TODO: make sure that the components are the right order of the pipeline
        self.components = components
        self.tuner_is_exist = any(isinstance(component, CollieComponents.TUNER.value) for component in self.components)
        self.mlflow_tags = mlflow_tags
        self.tracking_uri = tracking_uri
        self.description = description
        self.experiment_name = experiment_name
        
        if not self.tracking_uri:
            # TODO: use the better way
            self.tracking_uri = "./metadata/"
        self.mlflow_client = MlflowClient(tracking_uri=self.tracking_uri)

    @abstractmethod
    def orchestrate_pipeline(self) -> Any:
        raise NotImplementedError

    def run(self) -> Any:

        self.set_tracking_uri(self.tracking_uri)
        self.set_experiment(self.experiment_name)
        experiment_id = self.get_exp_id(self.experiment_name)

        with self.start_run(
            tags=self.mlflow_tags, 
            run_name="Orchestrator", 
            description=self.description, 
            experiment_id=experiment_id
        ):

            res = self.orchestrate_pipeline()
        return res

    def is_initialize_event_flavor(self, component: CollieComponentType) -> bool:
        if isinstance(component, CollieComponents.TRANSFORMER.value):
            return True
        return False

    def is_transformer_event_flavor(self, component: CollieComponentType) -> bool:
        if isinstance(component, CollieComponents.TRAINER.value) and not self.tuner_is_exist:
            return True
        if isinstance(component, CollieComponents.TUNER.value):
            return True
        return False

    def is_tuner_event_flavor(self, component: CollieComponentType) -> bool:
        if isinstance(component, CollieComponents.TRAINER.value):
            return True
        return False
    
    def is_trainer_event_flavor(self, component: CollieComponentType) -> bool:
        if isinstance(component, CollieComponents.EVALUATOR.value):
            return True
        return False

    def is_evaluator_event_flavor(self, component: CollieComponentType) -> bool:
        if isinstance(component, CollieComponents.PUSHER.value):
            return True
        return False
    
    def start_event(self) -> Event:
        """Initialize pipeline with an event."""
        return Event(
            type=EventType.INITAILIZE,
            payload=None
        )