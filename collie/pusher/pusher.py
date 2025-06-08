from typing import Literal, List

from collie.contracts.event import Event, EventHandler
from collie.contracts.mlflow import MLFlowComponentABC
from collie._common.types import EventType, PusherPayload
from collie._common.decorator import type_checker


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
            # The logic to push the model version to the target stage(OR deployment)
            pusher_event = self._handle(event)
            payload = self._get_evaluator_payload(pusher_event)

            version = self._get_version_to_transition(["Staging"])
            self.transition_model_version(
                registered_model_name=self.registered_model_name,
                version=version,
                desired_stage=self.target_stage,
                archive_existing_versions_at_stage=self.archive_existing_versions,
            )

            return Event(
                type=EventType.PUSHER_DONE,
                payload={"version_pushed": version, "stage": self.target_stage},
                context=event.context,
            )

    def _get_version_to_transition(
        self,
        stages: List[Literal["None", "Staging", "Production", "Archived"]],
    ) -> str:
        versions = self.get_latest_version(self.registered_model_name, stages=stages)
        return str(versions)

    @type_checker((PusherPayload,), "PusherPayload must be of type PusherPayload.")
    def _get_evaluator_payload(self, event: Event) -> PusherPayload:
        return event.payload