from typing import Union
from enum import Enum

from collie.core.transform.transform import Transformer
from collie.core.tuner.tuner import Tuner
from collie.core.trainer.trainer import Trainer
from collie.core.evaluator.evaluator import Evaluator
from collie.core.pusher.pusher import Pusher


class CollieComponents(Enum):

    TRAINER = Trainer
    TRANSFORMER = Transformer
    TUNER = Tuner 
    EVALUATOR = Evaluator
    PUSHER = Pusher


CollieComponentType = Union[
    Trainer, Transformer, Tuner, Evaluator, Pusher
]