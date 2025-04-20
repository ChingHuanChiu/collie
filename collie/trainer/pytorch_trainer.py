from typing import (
    Optional, 
    Tuple, 
    Dict,
    Any,
    List,
)
from abc import abstractmethod


from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch.amp import GradScaler, autocast
import torch

from collie.trainer.trainer import Trainer
from collie._common.decorator import type_checker
from collie._common.types import ComponentOutput
from collie._common.utils import get_logger
from collie.common.pytorch.callback.callback import Callback, _CallbackManager
from collie.common.pytorch.callback.model_checkpoint import ModelCheckpoint
from collie.common.pytorch.callback.earlystop import EarlyStopping

#TODO: Features to develop:
# 1.. Train with multiple GPUs 


logger = get_logger()


class _AbstractPytorchTrainer(Trainer):
    @abstractmethod
    def train_step(
        self,
        epoch_step: int,
        batch_data: torch.Tensor,
    ) ->  torch.Tensor: #loss

        raise NotImplementedError("Please implement the *train_step* method.")
        
    def validation_step(
        self,
        epoch_step: int,
        batch_data: torch.Tensor,
    ) -> Optional[torch.Tensor]: #loss
        
        return None

    @abstractmethod
    def create_train_val_dataloader(
        self, 
        outputs: ComponentOutput
    ) -> Tuple[DataLoader, Optional[DataLoader]]:

        raise NotImplementedError("Please implement the *create_train_val_dataloader* method.")
    
    @abstractmethod
    def configure_optimizers(
        self
    ) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Configure the optimizers and learning rate schedulers.

        Returns:
            A tuple of two objects. The first object is the optimizers and the second is the learning rate schedulers if necessary.

        Example:
            code block :: python

            def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initial_lr)
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    num_training_steps=self.num_training_steps
                )

                return optimizer, scheduler
        """
        raise NotImplementedError("Please implement the *configure_optimizers* method.")
    

class PytorchTrainer(_AbstractPytorchTrainer):
    def __init__(
        self, 
        model: nn.Module,
        epochs: int,
        device: Optional[str],
        use_amp: Optional[bool],
        topk_checkpoints: Optional[int],
        earlystop_patience_on_epoch: Optional[int] = None,
        accumulate_grad_batches: int = 1,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:

        super().__init__()

        self.model = model

        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.topk_checkpoints = 3 if topk_checkpoints is None else topk_checkpoints
        self.earlystop_patience_on_epoch = earlystop_patience_on_epoch

        DEFAULT_CALLBACK = [ModelCheckpoint(topk_checkpoints=self.topk_checkpoints)]
        if self.earlystop_patience_on_epoch:
            DEFAULT_CALLBACK.append(EarlyStopping(self.earlystop_patience_on_epoch, delta=0.0))
        
        self.callbacks = [] if callbacks is None else callbacks
        self.callbacks = self.callbacks + DEFAULT_CALLBACK
        self.accumulate_grad_batches = accumulate_grad_batches

        self.train_data_loader = None
        self.val_data_loader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.grad_scaler = None
        self.should_stop = False
        self.cb_manager = _CallbackManager(callbacks=self.callbacks)

        self.model = self.model.to(self.device)

    def train(self) -> Dict[str, Any]: 

        self.train_data_loader, self.val_data_loader = self._get_dataloader()
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        
        trainstep_per_epoch = len(self.train_data_loader)
        total_train_step = trainstep_per_epoch * self.epochs

        self.log_param("epoch", self.epochs)
        self.log_param("batch size", self.train_data_loader.batch_size)
        self.log_param("total training step", total_train_step)

        self.cb_manager.on_train_start(self)
        self.optimizer.zero_grad()
        for epoch_idx in tqdm(range(1, self.epochs + 1)):
            # Modified by Earlystop callback
            if self.should_stop:
                break

            self.cb_manager.on_epoch_start(self, epoch_idx)
            running_loss = 0
            self.model.train()
            for batch_idx, batch_data in enumerate(self.train_data_loader, start=1):
                self.cb_manager.on_batch_start(
                    batch_data=batch_data,
                    batch_step=batch_idx,
                    trainer=self
                )
                
                loss = self._get_train_step_result(epoch_idx, batch_data) / self.accumulate_grad_batches
                running_loss += loss.item()  * self.accumulate_grad_batches

                self._backward(loss=loss, batch_idx=batch_idx)
                # The learning rate scheduler is batch level.
                if self.lr_scheduler and batch_idx % self.accumulate_grad_batches == 0:
                    self.lr_scheduler.step()

                self.cb_manager.on_batch_end(
                    batch_step=batch_idx,
                    batch_data=batch_data,
                    batch_train_loss=running_loss,
                    trainer=self
                )

            epoch_loss = running_loss / trainstep_per_epoch
            
            self.log_metric("train loss", epoch_loss, step=epoch_idx)
            self._log_lr(epoch_step=epoch_idx)

            epoch_val_loss = self.validation_loop(epoch_step=epoch_idx)
            
            self.cb_manager.on_epoch_end(
                trainer=self,
                epoch_step=epoch_idx,
                epoch_train_loss=epoch_loss,
                epoch_val_loss=epoch_val_loss
            )
        
        # TODO: Fix the issue of loading the model from the following result model.
        self.log_model(self.model, model_type="pt")

        self.cb_manager.on_train_end(self)

        return {"model":self.model, "loss": epoch_loss}

    def validation_loop(self, epoch_step: int) -> Optional[float]:
        
        if self.val_data_loader is None:
            return None
        
        valstep_per_epoch = len(self.val_data_loader)

        self.model.eval()
        val_running_loss = 0
        with torch.no_grad():
            
            for val_batch_data in self.val_data_loader:
                
                val_loss = self.validation_step(epoch_step, val_batch_data)
                val_running_loss += val_loss.item()

            epoch_val_loss = val_running_loss / valstep_per_epoch
        
        self.log_metric("val_loss", epoch_val_loss, step=epoch_step)
        return epoch_val_loss

    def _backward(self, loss: torch.Tensor, batch_idx: int) -> None:

        def should_step_optimizer(batch_idx: int) -> bool:
            return batch_idx % self.accumulate_grad_batches == 0 or batch_idx == len(self.train_data_loader)

        if self.use_amp:
            if self.grad_scaler is None:
                self.grad_scaler = GradScaler(device=self.device)

            self.grad_scaler.scale(loss).backward()
            if should_step_optimizer(batch_idx):
                
                if self._has_invalid_gradients():
                    # gradient clipping
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.grad_scaler.step(self.optimizer)  
                self.grad_scaler.update() 
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if should_step_optimizer():
                if self._has_invalid_gradients(batch_idx):
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
    
    def _has_invalid_gradients(self) -> bool:
   
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.warning(f"Detected NaN or Inf gradients for parameter {param}. Zeroing out the gradients.")
                    # side effect
                    param.grad.zero_()
                    return True
        return False
            
    def _log_lr(self, epoch_step: int) -> None:
        
        lr = self.optimizer.param_groups[0]['lr'] if not self.lr_scheduler else self.lr_scheduler.get_last_lr()[0]
        self.log_metric("learning rate", lr, step=epoch_step)

    @type_checker((torch.Tensor,), 
                  "The return type of *train_step* method must be 'Tensor'.")
    def _get_train_step_result(self, epoch_step: int, batch_data: torch.Tensor):

        if self.use_amp:
            with autocast(device_type=self.device):
                loss = self.train_step(epoch_step, batch_data)
        else:
            loss = self.train_step(epoch_step, batch_data)

        return loss

    @type_checker((tuple,),
                  "The teturn type of *create_torch_dataloader* method must be 'Tuple'")
    def _get_dataloader(self):
        outputs: ComponentOutput = self.outputs
        return self.create_train_val_dataloader(outputs)
    
    def _get_optimizers(
        self
    ) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
        """
        Get the optimizer and learning rate scheduler from the 
        *configure_optimizers* method.
        """
        result = self.configure_optimizers()
        if isinstance(result, tuple):
            optimizers, lr_scheduler = result
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                raise TypeError(
                    f"learning rate scheduler must be type of \
                    torch.optim.lr_scheduler._LRScheduler "
                )
        else:
            optimizers, lr_scheduler = result, None
        return optimizers, lr_scheduler
        