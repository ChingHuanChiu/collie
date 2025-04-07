# Trainer
The trainer component is responsible for handling the training process. It receives the training data from the previous component, typically the `Transformer`. To access the output from the previous component, you need to retrieve it through the `outputs` argument.  
The output will be saved in the outputs argument of the next component with the key "Transformer",for example:  
 ```outputs = {"Trainer": model}```.

# Features
Currently, only `XGBTrainer` and `PytorchTrainer` are supported; other abstract classes, such as `SklearnTrainer`, are in development.

# Usage
Inherit from the `XGBTrainer` class or `PytorchTrainer` and override the relevant methods: 

* For `XGBTrainer`: override `fit_model`, 
* For `PytorchTrainer`: override `train_step` and `create_torch_dataloader` 
 
 
 See the following two examples:
 1. XGBRanker 
 ```python
 import xgboost as xgb

from collie import XGBTrainer
from ltr.common.strategy.spliter import DataSpliter


class LTRTrainer(XGBTrainer):

    def __init__(self, 
                 model_destination: str,
                 early_stopping_rounds: int,
                 train_ratio: int) -> None:
        
        super().__init__()
        self.model_destination = model_destination
        self.earlystop_round = early_stopping_rounds
        self.train_ratio = train_ratio

    def fit_model(self, outputs: Dict[str, Any]) -> xgb.XGBRanker:
        
        # Get the output of component from outputs argument
        examples = outputs["Transformer"]
        params = outputs["Tuner"]

        # Your model trianing code
        model = ....

        return model

 ```
 2. PytorchTrainer  
 ### NOTICE
* The return value of `train_step` must include both `loss` and `y_predict`. If `y_predict` is not needed, return `None` in its place.
* The return value of `create_torch_dataloader` must include both `train dataloader` and `validation dataloader`. If `validation dataloader` is not needed, return `None` in its place.
 ```python
 
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

import torch

from collie.trainer.common.metrics.base import MetricBase
from collie.trainer import PytorchTrainer


class EcomBertTrainer(PytorchTrainer):

    def __init__(self, 
                 model: Module, 
                 epochs: int, 
                 metric: MetricBase | None, 
                 optimizer: torch.optim.Optimizer, 
                 early_stop_step: Optional[int],
                 lr_scheduler: LRScheduler | None = None, 
                 device: str | None = None, 
                 using_amp: bool = False) -> None:
        
        super().__init__(model, 
                         epochs, 
                         metric, 
                         optimizer, 
                         lr_scheduler, 
                         device,
                         early_stop_step, 
                         using_amp,
                         )
        
    def train_step(self, 
                   x_batch: torch.Tensor, 
                   y_batch: torch.Tensor | None) -> Tuple[torch.Tensor]:
        
        outputs = self.model(x_batch)
        loss = outputs.loss

        return loss, None
    
    def create_torch_dataloader(self, outputs: Dict[str, Any]) -> Tuple[DataLoader | None]:
        
        train_data_loader = outputs["Transformer"]
        
        return train_data_loader, None
 
 ```