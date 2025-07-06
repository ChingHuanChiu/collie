from typing import (
    Optional,
    Dict,
    Union
)
from datetime import datetime


from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from collie.contracts.event import Event
from collie._common.types import EventType
from collie.contracts.orchestrator import OrchestratorABC
from collie._common.types import CollieComponents


class AirflowOrchestrator(OrchestratorABC):

    def __init__(
        self,
        dag_id: str,
        tracking_uri: str,
        components: Union[
            CollieComponents.TRAINER.value, 
            CollieComponents.TRANSFORMER.value,
            CollieComponents.TUNER.value,
            CollieComponents.EVALUATOR.value
        ],
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
            context["ti"].xcom_push(key="event", value=initial_event)

        start_task = PythonOperator(
            task_id="start",
            python_callable=start,
            provide_context=True,
            dag=self.dag
        )

        previous_task = start_task

        for idx, component in enumerate(self.components):
            
            component.mlflow_client = self.mlflow_client

            def _component_runner(comp):
                def _run(**context):
                    ti = context["ti"]
                    # incoming_event = ti.xcom_pull(task_ids=context["params"]["upstream"], key="event")
                    # TODO: the event data is from using mlflow.download_artifacts or download_model
                    incoming_event = ...
                    new_event = comp.run(incoming_event)
                    ti.xcom_push(key="event", value=new_event)
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
    

        # self.set_tracking_uri(self.tracking_uri)
        # self.set_experiment(self.experiment_name)
        # experiment_id = self.get_exp_id(self.experiment_name)

        # with self.start_run(
        #     tags={"component": "LocalPipeline"}, 
        #     run_name="Pipeline", 
        #     description=self.description, 
        #     experiment_id=experiment_id
        # ):

        #     for component in self.components:
        #         component.mlflow_client = self.mlflow_client
        #         _ = component.run()
        #     component.clear()