from enum import Enum

from collie.transform.transform import Transformer
from collie.trainer.trainer import Trainer
from collie.tuner.tuner import Tuner
from collie.evaluator.evaluator import Evaluator


class CollieComponents(Enum):

    TRAINER = Trainer
    TRANSFORMER = Transformer
    TUNER = Tuner 
    EVALUATOR = Evaluator