from typing import (
    Optional,
    Dict,
    Union,
    Any
)
from datetime import datetime
import io

import pandas as pd


from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from collie.contracts.event import Event
from collie.contracts.orchestrator import OrchestratorBase
from collie._common.types import (
    CollieComponentType,
    CollieComponents,
    TransformerPayload,
    TrainerPayload,
    TunerPayload,
    EvaluatorPayload,
    PusherPayload,
    EventType,
    TransformerArtifactPath,
    TrainerArtifactPath,
    TunerArtifactPath,
    EvaluatorArtifactPath,
    PusherArtifactPath
)
from collie._common.utils import get_logger


logger = get_logger()


class AirflowOrchestrator(OrchestratorBase):

    def __init__(
        self,
        dag_id: str,
        tracking_uri: str,
        components: CollieComponentType,
        mlflow_tags: Optional[Dict[str, str]] = None,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        default_args: Optional[Dict[str, str]] = None
    ) -> None:
        super().__init__(
            tracking_uri=tracking_uri, 
            components=components, 
            mlflow_tags=mlflow_tags, 
            experiment_name=experiment_name, 
            description=description
        )

        self.dag_id = dag_id
        self.default_args = default_args or {
            "owner": "airflow",
            "start_date": datetime.today()
        }

        self.dag = DAG(
            dag_id=self.dag_id,
            default_args=self.default_args,
            schedule_interval=None,
            catchup=False,
        )

    def run_pipeline(self) -> DAG:
        
        def start(**context):
            initial_event = Event(
                type=EventType.INITAILIZE, 
                payload=None
            )

            return initial_event

        start_task = PythonOperator(
            task_id="start",
            python_callable=start,
            provide_context=True,
            dag=self.dag
        )

        previous_task = start_task

        for idx, component in enumerate(self.components):
            
            component.mlflow_client = self.mlflow_client

            def _component_runner(component):
                def _run(**context):

                    if self.is_initialize_event_flavor(component):
                        initialize_event = Event(
                            type=EventType.INITAILIZE ,
                            payload=None
                        )

                        incoming_event = initialize_event

                    if self.is_transformer_event_flavor(component): 
                        transformer_payload = dict()
                        artifact_path: Dict[str, str] = TransformerArtifactPath.model_dump()
                        for data_type, data_artifact_path in artifact_path.items():
                            data_str = self.load_text(data_artifact_path)
                            transformer_payload[data_type] = pd.read_csv(io.StringIO(data_str))

                        transformer_event = Event(
                            type=EventType.DATA_READY,
                            payload=transformer_payload
                        )

                        incoming_event = transformer_event

                    elif self.is_tuner_event_flavor(component):
                        tuner_payload = dict()
                        artifact_path: str = TunerArtifactPath.hyperparameters
                        hyperparameters: Dict[str, Any] = self.load_dict(artifact_path)
                        # TODO: deal with the key using the enum method
                        tuner_payload["hyperparameters"] = hyperparameters["hyperparameters"]

                        tuner_event = Event(
                            type=EventType.DATA_READY,
                            payload=tuner_payload
                        )
                        
                        incoming_event = tuner_event
                    elif self.is_trainer_event_flavor(component):
                        trainer_payload = dict()
                        artifact_path: str = TrainerArtifactPath.model
                        model = self.load_model(artifact_path)
                        # TODO: deal with the key using the enum method
                        trainer_payload["model"] = model

                        trainer_event = Event(
                            type=EventType.DATA_READY,
                            payload=trainer_payload
                        )
                        incoming_event = trainer_event
                    elif self.is_evaluator_event_flavor(component):
                        eval_payload = dict()
                        artifact_path: str = EvaluatorArtifactPath.metrics
                        metrics = self.load_dict(artifact_path)
                        # TODO: deal with the key using the enum method
                        eval_payload["metrics"] = metrics["metrics"]

                        eval_event = Event(
                            type=EventType.DATA_READY,
                            payload=eval_payload
                        )
                        incoming_event = eval_event
                    else:
                        logger.WARNING(f"The current component type is passed because it is not supported: {type(component)}")
                        pass

                    _ = component.run(incoming_event)
                  
                return _run

            component_task = PythonOperator(
                task_id=f"component_{idx}",
                python_callable=_component_runner(component),
                provide_context=True,
                dag=self.dag
            )

            previous_task >> component_task
            previous_task = component_task
        return self.dag