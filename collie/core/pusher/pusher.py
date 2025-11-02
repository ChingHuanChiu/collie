from typing import Optional
from collie.contracts.event import (
    Event, 
    EventHandler, 
    EventType
)
from collie.contracts.mlflow import MLFlowComponentABC
from collie.core.models import PusherPayload
from collie._common.decorator import type_checker
from collie._common.exceptions import PusherError
from collie.core.enums.ml_models import MLflowModelStage


class Pusher(EventHandler, MLFlowComponentABC):
    def __init__(
        self,
        target_stage: MLflowModelStage = MLflowModelStage.PRODUCTION,
        archive_existing_versions: bool = True,
        description: Optional[str] = None,
        tags: Optional[dict] = None
    ) -> None:
        """
        Initializes the Pusher.

        Args:
            target_stage (Optional[MLflowModelStage], optional): The stage to transition the model to after evaluation. Defaults to None.
            archive_existing_versions (bool, optional): Whether to archive existing versions at the target stage. Defaults to True.
            description (Optional[str], optional): Description for the MLflow run. Defaults to None.
            tags (Optional[dict], optional): Tags to associate with the MLflow run. Defaults to None.

        """
        super().__init__()
        self._registered_model_name = None
        self.target_stage = target_stage
        self.archive_existing_versions = archive_existing_versions
        self.description = description
        self.tags = tags or {"component": "Pusher"}
    
    @property
    def registered_model_name(self) -> str:
        if not self._registered_model_name:
            raise PusherError("Registered model name is not set.")
        return self._registered_model_name
    
    @registered_model_name.setter
    def registered_model_name(self, name: str) -> None:
        self._registered_model_name = name

    def run(self, event: Event) -> Event:
        with self.start_run(
            tags=self.tags,
            run_name="Pusher",
            log_system_metrics=False,
            nested=True,
            description=self.description
        ):
            try:
                pusher_event = self._handle(event)
                payload = self._get_pusher_payload(pusher_event)
                pass_evaluation = event.context.get("pass_evaluation")
                
                if pass_evaluation:
                    model_uri = event.context.get('model_uri')
                    if not model_uri:
                        raise PusherError("model_uri not found in event context")
                    
                    version = self.register_model(
                        model_name=self.registered_model_name,
                        model_uri=model_uri
                    )
                    self.mlflow.log_param("model_registered", "true")
                    event.context.set("registered_version", str(version))
                    self.mlflow.log_param("model_version", version)
                    
                    self.mlflow.log_param("target_stage", self.target_stage.value)
                    self.transition_model_version(
                        registered_model_name=self.registered_model_name,
                        version=str(version),
                        desired_stage=self.target_stage,
                        archive_existing_versions_at_stage=self.archive_existing_versions,
                    )

                else:
                    self.mlflow.log_param("skip_reason", "evaluation_failed")
                    self.mlflow.log_param("model_registered", "false")

                return Event(
                    type=EventType.PUSHER_DONE,
                    payload=payload,
                    context=event.context,
                )

            except Exception as e:
                raise PusherError(f"Pusher failed with error: {e}")

    @type_checker((PusherPayload,), "PusherPayload must be of type PusherPayload.")
    def _get_pusher_payload(self, event: Event) -> PusherPayload:
        return event.payload