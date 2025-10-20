Contracts API Reference
=======================

This section documents the contract interfaces that define how components interact.

Usage Examples
--------------

Complete Component with MLflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python


   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score
   import matplotlib.pyplot as plt

  from collie import (
    Transformer,
    Trainer,
    Evaluator,
    TransformerPayload,
    TrainerPayload,
    EvaluatorPayload,
    Event
  )

   class DataTransformer(Transformer):
       def handle(self, event: Event) -> Event:
           # Load data
           df = pd.read_csv("data.csv")
           
           # Log data info as parameters
           self.mlflow.log_params({
               "n_samples": len(df),
               "n_features": len(df.columns) - 1,
               "data_source": "data.csv"
           })
           
           # Process data - prepare train/test split
           train_data = df.sample(frac=0.8, random_state=42)
           test_data = df.drop(train_data.index)
           
           return Event(
               payload=TransformerPayload(
                   train_data=train_data,
                   validation_data=None,
                   test_data=test_data
               )
           )

   class ModelTrainer(Trainer):
       def handle(self, event: Event) -> Event:
           train_data = event.payload.train_data
           X = train_data.drop("target", axis=1)
           y = train_data["target"]
           
           # Log hyperparameters
           params = {
               "n_estimators": 100,
               "max_depth": 10,
               "random_state": 42
           }
           self.mlflow.log_params(params)
           
           # Train model
           model = RandomForestClassifier(**params)
           model.fit(X, y)
           
           # Log training metrics
           train_score = model.score(X, y)
           self.mlflow.log_metric("train_accuracy", train_score)
           
           # Log feature importance
           importance = dict(zip(X.columns, model.feature_importances_))
           self.mlflow.log_dict(importance, "feature_importance.json")
           
           # Tag the run
           self.mlflow.set_tags({
               "model_type": "random_forest",
               "framework": "sklearn"
           })
           
           return Event(
               payload=TrainerPayload(
                   model=model,
               )
           )

   class ModelEvaluator(Evaluator):
       def handle(self, event: Event) -> Event:
           model = event.payload.model
           test_data = event.payload.test_data
           X_test = test_data.drop("target", axis=1)
           y_test = test_data["target"]
           
           # Evaluate
           y_pred = model.predict(X_test)
           accuracy = accuracy_score(y_test, y_pred)
           
           # Log metrics
           self.mlflow.log_metrics({
               "test_accuracy": accuracy,
               "n_test_samples": len(X_test)
           })
           
           # Create plot
           fig, ax = plt.subplots()
           # ... create confusion matrix plot ...
           plt.savefig("confusion_matrix.png")
           self.mlflow.log_artifact("confusion_matrix.png")
           plt.close()
           
           return Event(
               payload=EvaluatorPayload(
                   metrics=[{"test_accuracy": accuracy}],
                   is_better_than_production=accuracy > 0.85
               )
           )
