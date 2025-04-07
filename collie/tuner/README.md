# Tuner

The tuner component is responsible for finding the best hyperparameters. It receives the training data from the previous component, typically the `Transformer`. The output will be saved in the outputs argument of the next component with the key "Tuner",for example:  
 ```outputs = {"Tuner": best_hyper}```.

The tuner is implemented using the `hyperopt` package.

# Features
Currently, only `XGBTuner` are supported.

# Usage
Inherit from the `XGBTuner` class and override the `objective` method. You can receive the example data from the `outputs` argument of the objective method.

```python
from hyperopt import STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.model_selection import cross_val_score, GroupKFold

from collie import XGBTuner
from ltr.common.helper import (
    get_group_size, 
    train_xgb_model, 
    beta_pdf
)


class LTRTuner(XGBTuner):

    def __init__(self, 
                 hyper_space: Dict[str, Any], 
                 max_evals: int,
                 nkfold: int,
                 ) -> None:
        
        super().__init__(hyper_space=hyper_space, max_evals=max_evals)
        self.nkfold = nkfold

    def objective(self, outputs: Dict[str, Any], params: Dict[str, Any]):

        examples = outputs["Transformer"]

        target_col = "score"
        
        gkfold = GroupKFold(n_splits=self.nkfold)
    
        gs = gkfold.split(X=examples,
                          groups=examples.q.values)
        
        train_scores_list, val_scores_list = [], []
        
        for sub_train_idx, sub_val_idx in gs:
            
            sub_train = examples.iloc[sub_train_idx, :]
            sub_val = examples.iloc[sub_val_idx, :]

            sub_x_train, sub_y_train = \
                sub_train.drop([target_col, "q"], axis=1), sub_train[target_col]
            
            sub_x_val, sub_y_val = sub_val.drop([target_col, "q"], axis=1), sub_val[target_col]

            ranker = train_xgb_model(x_train=sub_x_train,
                                    y_train=sub_y_train,
                                    x_val=sub_x_val,
                                    y_val=sub_y_val,
                                    train_group_size=get_group_size(sub_train, "q"),
                                    val_group_size=get_group_size(sub_val, "q"),
                                    early_stopping_rounds=50,
                                    callbacks=[xgb.callback.LearningRateScheduler(beta_pdf(n_boosting_rounds=params["n_estimators"]))],
                                    params=params,
                                    verbose=False)

            score_result = ranker.evals_result()
            train_scores_list.append(np.mean(score_result['validation_1']['ndcg-']))
            val_scores_list.append(np.mean(score_result['validation_0']['ndcg-']))


        train_scores = np.mean(train_scores_list)
        val_scores = np.mean(val_scores_list)

        loss = -val_scores

        return {
                'loss': loss,
                'prams': params,
                'status': STATUS_OK
        }

```

# TODO:
* Encapsulate the PyTorch tuning logic.(PytorchTuner)