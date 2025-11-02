Getting Started
===============

This guide will help you get started with Collie, a lightweight MLOps framework.

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.10 or higher
* MLflow server (for tracking and model registry)

Install Collie
~~~~~~~~~~~~~~

.. code-block:: bash

   pip install collie-mlops

This will install Collie with all supported ML frameworks including scikit-learn, PyTorch, 
XGBoost, LightGBM, and Transformers.

Setting Up MLflow
~~~~~~~~~~~~~~~~~

Collie requires a running MLflow server. Start one locally:

.. code-block:: bash

   # Start MLflow server
   mlflow server --host 0.0.0.0 --port 5000

Or use an existing MLflow tracking server.

Your First Pipeline
-------------------

Let's create a simple ML pipeline using the Iris dataset.

Step 1: Create a Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Transformer handles data preprocessing:

.. code-block:: python

   from collie import Transformer
   from sklearn.datasets import load_iris
   import pandas as pd

   class IrisTransformer(Transformer):
       def handle(self, event):
           # Load data
           data = load_iris()
           df = pd.DataFrame(data.data, columns=data.feature_names)
           df['target'] = data.target
           
           # Log data statistics using MLflow
           self.mlflow.log_params({
               "n_samples": len(df),
               "n_features": len(data.feature_names)
           })
           
           # Log the input dataset
           self.log_pd_data(
               data=df,
               context="training",
               source="sklearn.datasets.load_iris"
           )
           
           # Return Event with TransformerPayload
           from collie import TransformerPayload, Event
           payload = TransformerPayload(
               train_data=df,
               extra_data={
                   "feature_names": list(data.feature_names),
                   "target_col": "target"
               }
           )
           return Event(payload=payload)
           )
           return Event(type=EventType.DATA_READY, payload=payload)

Step 2: Create a Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~

The Trainer handles model training:

.. code-block:: python

   from collie import Trainer
   from sklearn.ensemble import RandomForestClassifier

   class IrisTrainer(Trainer):
       def handle(self, event):
           # Get data from transformer
           df = event.payload.train_data
           feature_names = event.payload.get_extra("feature_names", [])
           target_col = event.payload.get_extra("target_col", "target")
           
           # Prepare training data
           X = df[feature_names]
           y = df[target_col]
           
           # Define hyperparameters
           params = {
               "n_estimators": 100,
               "max_depth": 10,
               "random_state": 42
           }
           
           # Log hyperparameters using MLflow
           self.mlflow.log_params(params)
           
           # Train model
           model = RandomForestClassifier(**params)
           model.fit(X, y)
           
           # Calculate and log training accuracy
           train_accuracy = model.score(X, y)
           self.mlflow.log_metric("train_accuracy", train_accuracy)
           
           # Log feature importance
           importance = dict(zip(feature_names, model.feature_importances_))
           self.mlflow.log_dict(importance, "feature_importance.json")
           
           # Return Event with TrainerPayload
           from collie import TrainerPayload, Event, EventType
           payload = TrainerPayload(model=model)
           return Event(type=EventType.TRAINING_DONE, payload=payload)

Step 3: Create an Orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Orchestrator coordinates all components:

.. code-block:: python

   from collie import Orchestrator

   # Create orchestrator with your components
   orchestrator = Orchestrator(
       components=[
           IrisTransformer(),
           IrisTrainer()
       ],
       tracking_uri="http://localhost:5000",
       registered_model_name="iris_classifier",
       experiment_name="iris_experiment"
   )

   # Run the pipeline
   orchestrator.run()

Step 4: View Results
~~~~~~~~~~~~~~~~~~~~

Open your MLflow UI to see the results:

.. code-block:: bash

   # MLflow UI should be available at
   http://localhost:5000

You'll see:

* Logged parameters (n_samples, n_features, hyperparameters)
* Logged metrics (train_accuracy)
* Logged artifacts (feature_importance.json)
* Registered model (iris_classifier)

Complete Example
----------------

Here's the complete working example:

.. code-block:: python

   from collie import Transformer, Trainer, Orchestrator
   from collie import TransformerPayload, TrainerPayload, Event, EventType
   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier
   import pandas as pd

   class IrisTransformer(Transformer):
       def handle(self, event):
           data = load_iris()
           df = pd.DataFrame(data.data, columns=data.feature_names)
           df['target'] = data.target
           
           self.mlflow.log_params({
               "n_samples": len(df),
               "n_features": len(data.feature_names)
           })
           
           self.mlflow.log_pd_data(
               data=df,
               context="training",
               source="sklearn.datasets.load_iris"
           )
           
           payload = TransformerPayload(
               train_data=df,
               extra_data={
                   "feature_names": list(data.feature_names),
                   "target_col": "target"
               }
           )
           return Event(type=EventType.DATA_READY, payload=payload)

   class IrisTrainer(Trainer):
       def handle(self, event):
           df = event.payload.train_data
           feature_names = event.payload.get_extra("feature_names", [])
           target_col = event.payload.get_extra("target_col", "target")
           
           X = df[feature_names]
           y = df[target_col]
           
           params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
           self.mlflow.log_params(params)
           
           model = RandomForestClassifier(**params)
           model.fit(X, y)
           
           train_accuracy = model.score(X, y)
           self.mlflow.log_metric("train_accuracy", train_accuracy)
           
           importance = dict(zip(feature_names, model.feature_importances_))
           self.mlflow.log_dict(importance, "feature_importance.json")
           
           payload = TrainerPayload(model=model)
           return Event(type=EventType.TRAINING_DONE, payload=payload)

   # Run the pipeline
   orchestrator = Orchestrator(
       components=[IrisTransformer(), IrisTrainer()],
       tracking_uri="http://localhost:5000",
       registered_model_name="iris_classifier",
       experiment_name="iris_experiment"
   )
   orchestrator.run()

Next Steps
----------

Now that you have a basic pipeline running, you can:

1. **Add Evaluation** - Create an Evaluator to assess model performance
2. **Add Tuning** - Create a Tuner for hyperparameter optimization
3. **Add Deployment** - Create a Pusher to deploy models
4. **Customize Components** - Extend components for your use case

See the :doc:`core_concepts` guide for more details.

Troubleshooting
---------------

MLflow Connection Error
~~~~~~~~~~~~~~~~~~~~~~~

**Error:** ``ConnectionError: Cannot connect to MLflow server``

**Solution:** Make sure MLflow server is running:

.. code-block:: bash

   mlflow server --host 0.0.0.0 --port 5000

Import Error
~~~~~~~~~~~~

**Error:** ``ModuleNotFoundError: No module named 'collie'``

**Solution:** Install collie-mlops (not collie):

.. code-block:: bash

   pip install collie-mlops

Model Registration Failed
~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:** ``Model registration failed``

**Solution:** Ensure ``registered_model_name`` is set in Orchestrator:

.. code-block:: python

   orchestrator = Orchestrator(
       components=[...],
       tracking_uri="http://localhost:5000",
       registered_model_name="my_model"  # Required for registration
   )

Get Help
--------

* GitHub Issues: https://github.com/ChingHuanChiu/collie/issues
* Documentation: https://github.com/ChingHuanChiu/collie/blob/main/README.md
* Examples: https://github.com/ChingHuanChiu/collie/tree/main/example
