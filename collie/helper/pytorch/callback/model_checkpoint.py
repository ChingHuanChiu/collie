from typing import Optional
from glob import glob
import os
import shutil

import torch

from collie.helper.pytorch.callback.callback import Callback
from collie._common.utils import get_logger
from collie import trainer

logger = get_logger()

class ModelCheckpoint(Callback):

    def __init__(self, topk_checkpoints: int):
        super().__init__()
        self.topk_checkpoints = topk_checkpoints
        self._best_checkpoints = [] # save as (loss, epoch_idx, checkpoint_path)
        
        self.parent_dir = "./.checkpoint/"
        if os.path.exists(self.parent_dir):
            shutil.rmtree(self.parent_dir)
            logger.info(f"Directory {self.parent_dir} has been removed.")

    def on_epoch_end(
        self,
        trainer: "trainer.PytorchTrainer", 
        epoch_step: int, 
        epoch_train_loss: float, 
        epoch_val_loss: Optional[float]
    ) -> None:
        
        current_loss = epoch_val_loss if epoch_val_loss else epoch_train_loss
        if self._should_save_checkpoint(current_loss):
            self._save_checkpoint(
                trainer=trainer,
                epoch_step=epoch_step,
                epoch_loss=epoch_train_loss,
                loss_for_ckpt=current_loss
            )
    def on_train_end(self, trainer):
        for ckpt in glob(f"{self.parent_dir}/*.pt"):
            trainer.log_artifact(ckpt, "checkpoints")

    def _should_save_checkpoint(self, current_loss: float) -> bool:
        """
        Returns a boolean indicating if the current checkpoint should be saved.

        The current checkpoint should be saved if the topk_checkpoints list is not full.
        Otherwise, the current checkpoint should be saved if its loss is better than the
        loss of the worst checkpoint in the topk_checkpoints list.

        Args:
            current_loss (float): The loss of the current checkpoint.

        Returns:
            bool: True if the current checkpoint should be saved, False otherwise.
        """
        if len(self._best_checkpoints) < self.topk_checkpoints:
            # If the topk_checkpoints list is not full, save the current checkpoint.
            return True

        worst_loss = max(self._best_checkpoints, key=lambda x: x[0])[0]
        # If the current checkpoint has a better loss than the worst checkpoint in the
        # topk_checkpoints list, save the current checkpoint.
        return current_loss < worst_loss
    
    def _save_checkpoint(
        self,
        trainer: "trainer.PytorchTrainer", 
        epoch_step: int, 
        epoch_loss: float,
        loss_for_ckpt: float
    ) -> None:

        checkpoint_path = self.parent_dir + f'model_epoch{epoch_step}.pt'

        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)

        # https://zhuanlan.zhihu.com/p/136902153
        checkpoint = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.lr_scheduler.state_dict() if trainer.lr_scheduler else None,
            "epoch": epoch_step,
            "loss": epoch_loss
        }

        if len(self._best_checkpoints) < self.topk_checkpoints:
            self._best_checkpoints.append((loss_for_ckpt, epoch_step, checkpoint_path))
        else:
            worst_idx, (worst_loss, _, worst_path) = max(enumerate(self._best_checkpoints), key=lambda x: x[1][0])

            os.remove(worst_path)  
            self._best_checkpoints[worst_idx] = (loss_for_ckpt, epoch_step, checkpoint_path)

        self._best_checkpoints.sort(key=lambda x: x[0])
        torch.save(checkpoint, f'{checkpoint_path}')