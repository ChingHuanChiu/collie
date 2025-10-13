from typing import (
    Optional,
    Dict,
)

from collie.contracts.orchestrator import OrchestratorBase
from collie.core.enums.components import CollieComponentType
from collie._common.utils import get_logger

logger = get_logger()


class Orchestrator(OrchestratorBase):

    def __init__(
        self,
        components: CollieComponentType,
        tracking_uri: str,
        registered_model_name: str,
        mlflow_tags: Optional[Dict[str, str]] = None,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            components=components,
            tracking_uri=tracking_uri,
            mlflow_tags=mlflow_tags,
            experiment_name=experiment_name,
            description=description
        )
        self.registered_model_name = registered_model_name

    def orchestrate_pipeline(self) -> None:
        """Run the pipeline sequentially without Airflow."""

        logger.info("Pipeline started.")
        incoming_event = self.initialize_event()

        for idx, component in enumerate(self.components):
            logger.info(f"Running component {idx}: {type(component).__name__}")
            component.mlflow_config = self.mlflow_config
            if hasattr(component, "_registered_model_name"):
                component.registered_model_name = self.registered_model_name
        
            incoming_event = component.run(incoming_event)

            logger.info(f"Component {idx} finished: {type(component).__name__}")