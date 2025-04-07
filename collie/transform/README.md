# Transformer
The transformer is responsible for preprocessing the data, which will serve as the training data for the next component. The output will be saved in the outputs argument of the next component with the key "Transformer",for example:  
 ```outputs = {"Transformer": result_data}```.

# Usage
Inherit from the `Transformer` class and override the `transform` method to return the trainable data. See the following two examples:
1.   Machine Learning
```python
from collie import Transformer

class ExampleTransformer(Transformer):
    # 'example' is your raw data. 
    def __init__(self, examples: Any) -> None:
        super().__init__(examples)


    def transform(self) -> Any:
    
        # Your process logic
        process_data = ...

        return process_data

```
2. Pytorch  
```python
from collie import Transformer
from torch.utils.data import Dataset, DataLoader

class EcombertTransformer(Transformer):

    def __init__(self, 
                 examples: str,
                 pretrain_model: str,
                 token_max_length: int,
                 mlm_probability: float ,
                 batch_size: int
                 ) -> None:
        super().__init__(examples)

        self.pretrain_model = pretrain_model
        self.token_max_length = token_max_length
        self.mlm_probability = mlm_probability
        self.batch_size = batch_size

    def transform(self) -> DataLoader:

        tokenizer = BertTokenizerFast.from_pretrained(self.pretrain_model)

        train_dataset = EcomBertDataset(train_data_path=self.examples, 
                                        tokenizer=tokenizer,
                                        max_length=self.token_max_length)
        
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                    mlm=True, 
                                                    mlm_probability=self.mlm_probability)
        
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=collator)

        return train_data_loader


class EcomBertDataset(Dataset):

    def __init__(self, 
                 train_data_path: str,
                 tokenizer,
                 max_length: int):
        
        with open(train_data_path, 'r') as f:
            self.source = list(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        text = self.source[idx].replace('\n','')
        encoded = self.tokenizer(text=text, truncation=True, max_length=self.max_length)
        return encoded

```