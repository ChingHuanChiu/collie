from typing import Optional, Dict, Union
from collections import OrderedDict

import torch


def load_pytorch_model_checkpoint(
    ckpt_path: str, 
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    is_ddp_model: bool = False
) -> Dict[str, Union[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]]:
    """
    Loads a PyTorch model checkpoint and optionally restores the optimizer and learning rate scheduler states.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state dict into.
        optimizer (Optional[torch.optim.Optimizer], optional): Optimizer to restore its state. Default is None.
        lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): Learning rate scheduler to restore its state. Default is None.
        is_ddp_model (bool, optional): Whether the model is wrapped in DistributedDataParallel. Default is False.

    Returns:
        Dict[str, Union[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int]]: 
            A dictionary containing the restored model, optimizer, learning rate scheduler (if any), and the epoch.
    """

    res_dict = dict()

    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['model_state_dict']
    
    # Handle DDP (DistributedDataParallel) model
    if is_ddp_model:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # remove 'module.' of DataParallel/DistributedDataParallel
            name = k[7:] 
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    res_dict["model"] = model.eval()
    res_dict["epoch"] = checkpoint.get("epoch", "")

    if optimizer :
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        res_dict["optimizer"] = optimizer

    if 'lr_scheduler_state_dict' in checkpoint and lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict']) 
        res_dict["lr_scheduler"] = lr_scheduler

    return res_dict