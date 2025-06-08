from typing import Optional, List, Literal

import torch

from collie import trainer


class Callback:

    def on_train_start(
        self,
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        ...

    def on_train_end(
        self,
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        ...

    def on_epoch_start(
        self,
        epoch_step: int,
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        ...

    def on_epoch_end(
        self, 
        epoch_step: int,
        epoch_train_loss: float,
        epoch_val_loss: Optional[float],
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        ...
    
    def on_batch_start(
        self, 
        batch_step: int, 
        batch_data: torch.Tensor,
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        ...

    def on_batch_end(
        self, 
        batch_step: int, 
        batch_data: torch.Tensor,
        batch_train_loss: float,
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        ...


class _CallbackManager:
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def on_train_start(
        self,
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        
        self._execute_callbacks(
            "on_train_start",
            trainer=trainer
        )
        
    def on_train_end(
        self,
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        
        self._execute_callbacks(
            "on_train_end",
            trainer=trainer
        )

    def on_epoch_start(
        self, 
        trainer: "trainer.PytorchTrainer", 
        epoch_step: int
    ) -> None:

        self._execute_callbacks(
            "on_epoch_start",
            trainer=trainer,
            epoch_step=epoch_step
        )

    def on_epoch_end(
        self, 
        trainer: "trainer.PytorchTrainer", 
        epoch_step: int,
        epoch_train_loss: float,
        epoch_val_loss: Optional[float],
    ) -> None:
         
        self._execute_callbacks(
            "on_epoch_end",
            trainer=trainer,
            epoch_step=epoch_step,
            epoch_train_loss=epoch_train_loss,
            epoch_val_loss=epoch_val_loss
        )
         
    def on_batch_start(
        self,
        batch_step: int, 
        batch_data: torch.Tensor,
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        
        self._execute_callbacks(
            "on_batch_start",
            trainer=trainer,
            batch_step=batch_step,
            batch_data= batch_data
        )

    def on_batch_end(
        self,
        batch_step: int, 
        batch_data: torch.Tensor,
        batch_train_loss: float,
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        
        self._execute_callbacks(
            "on_batch_end",
            trainer=trainer,
            batch_step=batch_step,
            batch_data= batch_data,
            batch_train_loss=batch_train_loss
        )

    def _execute_callbacks(
        self, 
        method_name: Literal[
            "on_train_start",
            "on_train_end",
            "on_epoch_start", 
            "on_epoch_end", 
            "on_batch_start", 
            "on_batch_end"], 
        *args, **kwargs
    ) -> None:
       
        if self.callbacks is None:
            return
        
        for callback in self.callbacks:
            # Call the method specified by method_name on the callback
            getattr(callback, method_name)(*args, **kwargs)