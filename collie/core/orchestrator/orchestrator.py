from typing import (
    Optional,
    Dict,
    Any
)
import io
import pandas as pd

from collie.contracts.event import Event, EventType
from collie.contracts.orchestrator import OrchestratorBase
from collie.core.types import CollieComponentType
from collie.core.models import (
    TrainerArtifactPath,
    TransformerArtifactPath,
    TunerArtifactPath,
    EvaluatorArtifactPath,
    PusherArtifactPath
)
from collie._common.utils import get_logger


logger = get_logger()


class Orchestrator(OrchestratorBase):

    def __init__(
        self,
        tracking_uri: str,
        components: CollieComponentType,
        mlflow_tags: Optional[Dict[str, str]] = None,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            tracking_uri=tracking_uri,
            components=components,
            mlflow_tags=mlflow_tags,
            experiment_name=experiment_name,
            description=description
        )

    def orchestrate_pipeline(self) -> None:
        """Run the pipeline sequentially without Airflow."""

        logger.info("Pipeline started.")
        incoming_event = self.start_event()

        for idx, component in enumerate(self.components):
            logger.info(f"Running component {idx}: {type(component).__name__}")
            component.mlflow_client = self.mlflow_client
            incoming_event = self.run_component(component, incoming_event)

        logger.info("Pipeline finished successfully.")

    def run_component(
        self, 
        component: CollieComponentType, 
        incoming_event: Event
    ) -> Event:
        """Prepare the payload for each component type and run it."""

        if self.is_initialize_event_flavor(component):
            incoming_event = Event(
                type=EventType.INITAILIZE,
                payload=None
            )

        elif self.is_transformer_event_flavor(component):
            transformer_payload = {}
            artifact_path: Dict[str, str] = TransformerArtifactPath.model_dump()
            for data_type, data_artifact_path in artifact_path.items():
                data_str = self.load_text(data_artifact_path)
                # TODO: use the mlflow api to load the data
                transformer_payload[data_type] = pd.read_csv(io.StringIO(data_str))

            incoming_event = Event(
                type=EventType.DATA_READY,
                payload=transformer_payload
            )

        elif self.is_tuner_event_flavor(component):
            artifact_path: str = TunerArtifactPath.hyperparameters
            hyperparameters: Dict[str, Any] = self.load_dict(artifact_path)
            tuner_payload = {"hyperparameters": hyperparameters["hyperparameters"]}

            incoming_event = Event(
                type=EventType.DATA_READY,
                payload=tuner_payload
            )

        elif self.is_trainer_event_flavor(component):
            artifact_path: str = TrainerArtifactPath.model
            model = self.load_model(artifact_path)
            trainer_payload = {"model": model}

            incoming_event = Event(
                type=EventType.DATA_READY,
                payload=trainer_payload
            )

        elif self.is_evaluator_event_flavor(component):
            artifact_path: str = EvaluatorArtifactPath.metrics
            metrics = self.load_dict(artifact_path)
            eval_payload = {"metrics": metrics["metrics"]}

            incoming_event = Event(
                type=EventType.DATA_READY,
                payload=eval_payload
            )

        else:
            raise ValueError(f"Unsupported component type: {type(component)}")

        return component.run(incoming_event)
