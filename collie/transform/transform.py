from abc import abstractmethod
from typing import Any

from collie.abstract.mlflow import MLFlowComponentABC
from collie._common.mixin import OutputMixin
from collie._common.types import ComponentOutput


class Transformer(MLFlowComponentABC, OutputMixin):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def transform(self) -> Any:
        raise NotImplementedError("Please implement the **transform** method.")
    
    def run(self) -> None:
        """
        Run the transformer component.

        This method starts a new MLflow run, transforms the input data,
        logs metrics, and sets the outputs.
        """
        with self.start_run(
            tags={"component": "Transformer"},
            run_name="Transformer",
            log_system_metrics=True,
            nested=True,
        ):
            transformed_data = self.transform()

            self.outputs: ComponentOutput = {
                "Transformer": transformed_data
            }