from typing import Any, Dict
from abc import abstractmethod

from collie.abstract.mlflow import MLFlowComponentABC
from collie._common.mixin import OutputMixin
from collie._common.types import ComponentOutput


class Trainer(MLFlowComponentABC, OutputMixin):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        raise NotImplementedError("Please implement the **train** method.")
    
    def run(self) -> None:
        """
        Run the trainer component.

        This method starts a new MLflow run, trains the model,
        logs metrics, and sets the outputs.
        """
        with self.start_run(
            run_name="Trainer", 
            tags={"component": "Trainer"},
            log_system_metrics=True, 
            nested=True
        ):
            train_results = self.train()

            self.outputs: ComponentOutput = {
                "Trainer": train_results.get("model")
            }

            model_loss = train_results.get("loss")
            if model_loss is not None:
                self.outputs.update({"ModelLoss": model_loss})