"""
Module Description:
...

Author: ChingHuanChiu
Email: "stevenchiu@example.com"
"""

__author__ = "ChingHuanChiu"
__email__ = "stevenchiu@example.com"
__version__ = "1.0.0"

from collie.transform.transform import Transformer
from collie.tuner.xgb_tuner import XGBTuner
# from collie.trainer.pytorch_trainer import PytorchTrainer
from collie.trainer.xgb_trainer import XGBTrainer
from collie.evaluator.evaluator import Evaluator

#TODO:
#1. Define the custom type for example
#2. Rewrite pytorch trainer.