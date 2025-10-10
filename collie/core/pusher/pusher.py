from typing import Literal, List

from collie.contracts.event import (
    Event, 
    EventHandler, 
    EventType
)
from collie.contracts.mlflow import MLFlowComponentABC
from collie.core.models import PusherPayload
from collie._common.decorator import type_checker
from collie._common.exceptions import PusherError


class Pusher(EventHandler, MLFlowComponentABC):
    def __init__(
        self,
        registered_model_name: str,
        target_stage: Literal["Production", "Staging"] = "Production",
        archive_existing_versions: bool = True,
    ) -> None:
        super().__init__()
        self.registered_model_name = registered_model_name
        self.target_stage = target_stage
        self.archive_existing_versions = archive_existing_versions

    def run(self, event: Event) -> Event:
        with self.start_run(
            tags={"component": "Pusher"},
            run_name="Pusher",
            log_system_metrics=False,
            nested=True,
        ):
            try:
                # The logic to push the model version to the target stage(OR deployment)
                pusher_event = self._handle(event)
                payload = self._get_pusher_payload(pusher_event)

                version = self._get_version_to_transition(["Staging"])
                self.transition_model_version(
                    registered_model_name=self.registered_model_name,
                    version=version,
                    desired_stage=self.target_stage,
                    archive_existing_versions_at_stage=self.archive_existing_versions,
                )

                return Event(
                    type=EventType.PUSHER_DONE,
                    payload=payload,
                    context=event.context,
                )

            except Exception as e:
                raise PusherError(f"Pusher failed with error: {e}")

    def _get_version_to_transition(
        self,
        stages: List[Literal["None", "Staging", "Production", "Archived"]],
    ) -> str:
        """
        Retrieves the latest version number of a model in the specified stages.

        Args:
            stages (List[Literal["None", "Staging", "Production", "Archived"]]): 
                The stages in which to search for the latest version.

        Returns:
            str: The latest version number of the model in the specified stages.

        Raises:
            PusherError: If no versions are found in the specified stages.
        """
        version = self.get_latest_version(self.registered_model_name, stages=stages)
        if not version:
            raise PusherError(
                f"No model versions found in stages {stages} for model {self.registered_model_name}"
            )
        return version

    @type_checker((PusherPayload,), "PusherPayload must be of type PusherPayload.")
    def _get_pusher_payload(self, event: Event) -> PusherPayload:
        return event.payload