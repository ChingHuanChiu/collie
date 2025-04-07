# Evaluator
The evaluator component is responsible for evaluate the models. It receives the training data from the previous component, typically the `Trainer`. The output will be saved in the outputs with the key "Evaluator",for example:  
 ```outputs = {"Evaluator": result}```.

# Usage
Inherit from the `Evaluator` class and override the `evaluate` method.  
See the following example:

```python
from collie import Evaluator

class ExampleEvaluator(Evaluator):

    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, outputs: Dict[str, Any]) -> None:

        model = outputs["Trainer"]
        #Your evaluation logic
        ....
```