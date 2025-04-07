from typing import (
    NewType, 
    Dict, 
    Union,
    Any
)

from pydantic import BaseModel


class EvaluatorDataModel(BaseModel):
    data: Dict[str, Union[int, Union[int, float]]]


ComponentOutput = NewType("ComponentOutput", Dict[str, Any])