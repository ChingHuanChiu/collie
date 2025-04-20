from typing import Optional

from collie.common.pytorch.callback.callback import Callback
from collie import trainer
from collie._common.utils import get_logger

logger = get_logger()


class EarlyStopping(Callback):

    def __init__(
        self, 
        patience_on_epoch: int, 
        delta: float = 0.0, 
        monitor: str = "val_loss"
    ):
        super().__init__()

        self.patience_on_epoch = patience_on_epoch
        self.delta = delta
        self.monitor = monitor
        self.best_score = float('inf')  
        self.wait = 0  
        self.stopped_epoch = 0  
        self.early_stop = False 

    def on_epoch_end(
        self, 
        epoch_step: int, 
        epoch_train_loss: float, 
        epoch_val_loss: Optional[float], 
        trainer: "trainer.PytorchTrainer"
    ) -> None:
        
        if self.monitor == "val_loss" and epoch_val_loss is not None:
            score = epoch_val_loss
        elif self.monitor == "train_loss":
            score = epoch_train_loss
        else:
            return  

       
        if self.best_score - score > self.delta:  
            self.best_score = score
            self.wait = 0  
        else:
            self.wait += 1  

        if self.wait >= self.patience_on_epoch:
            self.early_stop = True
            self.stopped_epoch = epoch_step
            trainer.should_stop = True
            logger.info(f"Epoch {epoch_step}: Early stopping triggered (patience {self.patience_on_epoch} epochs).")