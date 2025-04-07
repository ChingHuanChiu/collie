from typing import Dict, Any, Self
from collections import defaultdict


class OutputMixin:
    """
    A mixin class that provides a way to set and retrieve outputs from a component.
    """

    _outputs = defaultdict(lambda: None)

    @property
    def outputs(self) -> Dict[str, Any]:

        return self._outputs
    
    @outputs.setter
    def outputs(self, values: Dict[str, Any]) -> None:
        
        if not isinstance(values, dict):
            raise TypeError("Value must be type of dictionary.")
        self._outputs.update(values)
    
    @classmethod
    def clear(cls) -> None:
        cls._outputs.clear()