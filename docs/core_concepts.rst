Core Concepts
=============

Understanding Collie's Architecture
------------------------------------

Collie is built on three core principles:

1. **Modular Components** - Each stage of the ML pipeline is a separate component
2. **Event-Driven** - Components communicate through events
3. **MLflow Integration** - Deep integration for tracking and model management

Component Lifecycle
-------------------

Every Collie component follows this lifecycle:

.. code-block:: text

   1. Initialize → 2. Execute → 3. Log to MLflow → 4. Emit Event → 5. Next Component

The Pipeline Flow
-----------------

A typical Collie pipeline:

.. code-block:: text

   Raw Data → Transformer → Trainer → Tuner → Evaluator → Pusher → Deployed Model
                   ↓            ↓        ↓         ↓          ↓
                MLflow      MLflow   MLflow    MLflow     MLflow
              (data log)  (params) (trials)  (metrics)  (register)

Components in Detail
--------------------

Transformer
~~~~~~~~~~~

**Purpose:** Data preprocessing and feature engineering

**Responsibilities:**
- Load and clean data
- Feature engineering
- Data validation
- Train/test split

**MLflow Usage:**
- Log input datasets
- Log transformation parameters
- Log data statistics

**Example:**

.. code-block:: python

   from collie import Event
   from collie.core import TransformerPayload
   
   class MyTransformer(Transformer):
       def handle(self, event: Event) -> Event:
           # Load data
           df = pd.read_csv("data.csv")
           
           # Log input data
           self.mlflow.log_input_data(
               data=df,
               context="training",
               source="data.csv"
           )
           
           # Feature engineering
           df['new_feature'] = df['feature1'] * df['feature2']
           
           # Log stats
           self.mlflow.log_params({
               "n_samples": len(df),
               "n_features": len(df.columns)
           })
           
           return Event(
               payload=TransformerPayload(
                   train_data=df,
                   validation_data=None,
                   test_data=None
               )
           )

Trainer
~~~~~~~

**Purpose:** Model training

**Responsibilities:**
- Train machine learning models
- Log hyperparameters
- Log training metrics
- Save model artifacts

**MLflow Usage:**
- Log hyperparameters
- Log training metrics (loss, accuracy, etc.)
- Log model automatically

**Example:**

.. code-block:: python

   from collie import Event
   from collie.core import TrainerPayload
   
   class MyTrainer(Trainer):
       def handle(self, event: Event) -> Event:
           train_data = event.payload.train_data
           X = train_data.drop("target", axis=1)
           y = train_data["target"]
           
           # Log hyperparameters
           self.mlflow.log_params({
               "learning_rate": 0.01,
               "batch_size": 32
           })
           
           # Training loop
           for epoch in range(100):
               loss = train_one_epoch(model, X, y)
               self.mlflow.log_metric("loss", loss, step=epoch)
           
           # Return model in payload
           return Event(
               payload=TrainerPayload(model=model)
           )

Tuner
~~~~~

**Purpose:** Hyperparameter optimization

**Responsibilities:**
- Search hyperparameter space
- Track trials
- Select best parameters

**MLflow Usage:**
- Create nested runs for each trial
- Log trial parameters and metrics
- Log best parameters

**Example:**

.. code-block:: python

   from collie import Event
   from collie.core import TunerPayload
   
   class MyTuner(Tuner):
       def handle(self, event: Event) -> Event:
           best_score = 0
           best_params = {}
           
           for params in param_grid:
               # Each trial gets its own run
               with self.mlflow.start_run(nested=True):
                   self.mlflow.log_params(params)
                   
                   score = evaluate(params)
                   self.mlflow.log_metric("cv_score", score)
                   
                   if score > best_score:
                       best_score = score
                       best_params = params
           
           self.mlflow.log_dict(best_params, "best_params.json")
           
           return Event(
               payload=TunerPayload(
                   hyperparameters=best_params,
                   train_data=event.payload.train_data,
                   validation_data=event.payload.validation_data,
                   test_data=event.payload.test_data
               )
           )

Evaluator
~~~~~~~~~

**Purpose:** Model evaluation

**Responsibilities:**
- Evaluate model performance
- Compare with baseline/production models
- Generate evaluation reports
- Decide if model should be deployed

**MLflow Usage:**
- Log evaluation metrics
- Log evaluation plots
- Load production model for comparison

**Example:**

.. code-block:: python

   from collie import Event
   from collie.core import EvaluatorPayload
   from collie.core.enums.ml_models import MLflowModelStage, ModelFlavor
   
   class MyEvaluator(Evaluator):
       def handle(self, event: Event) -> Event:
           model = event.payload.model
           test_data = event.payload.test_data
           X_test = test_data.drop("target", axis=1)
           y_test = test_data["target"]
           
           # Evaluate
           y_pred = model.predict(X_test)
           experiment_accuracy = accuracy_score(y_test, y_pred)
           
           # Log metrics
           self.mlflow.log_metric("experiment_accuracy", experiment_accuracy)
           
           # Compare with production
           prod_model = self.load_latest_model(
               model_name=self.registered_model_name,
               stage=MLflowModelStage.PRODUCTION,
               flavor=ModelFlavor.SKLEARN
           )
           
           if prod_model is not None:
               production_accuracy = prod_model.score(X_test, y_test)
               is_better = experiment_accuracy > production_accuracy
           else:
               production_accuracy = 0
               is_better = True
           
           self.mlflow.log_param(
               "promotion_decision",
               "promote" if is_better else "keep_current"
           )
           
           return Event(
               payload=EvaluatorPayload(
                   metrics=[
                       {
                           "experiment_accuracy": experiment_accuracy,
                           "production_accuracy": production_accuracy,
                           "accuracy_improvement": experiment_accuracy - production_accuracy
                       }
                   ],
                   is_better_than_production=is_better
               )
           )

Pusher
~~~~~~

**Purpose:** Model deployment to production environments

**Responsibilities:**
- Deploy models to external services (APIs, Elasticsearch, etc.)
- Transition model stages in MLflow
- Handle deployment configurations
- Log deployment metadata

**MLflow Usage:**
- Transition model stages (Staging → Production)
- Log deployment parameters and tags

**Example: Deploy to External Service**

.. code-block:: python

   from collie import Event, Pusher
   from collie.core import PusherPayload
   import requests
   
   class ModelDeploymentPusher(Pusher):
       def handle(self, event: Event) -> Event:
           # Get model from payload
           model = event.payload.model
           
           # Example: Deploy to Elasticsearch or REST API
           deployment_endpoint = "https://your-ml-engine.com/api/models"
           
           try:
               # Serialize and post model
               response = requests.post(
                   deployment_endpoint,
                   json={
                       "model_name": "iris_classifier",
                       "model_data": serialize_model(model),
                       "metadata": event.payload.get_extra("model_metadata", {})
                   }
               )
               
               # Log deployment info
               self.mlflow.log_params({
                   "deployment_endpoint": deployment_endpoint,
                   "deployment_status": response.status_code
               })
               
               return Event(
                   payload=PusherPayload(
                       model_uri=f"external://{response.json()['model_id']}",
                       status="deployed",
                       model_version="1.0",
                       extra_data={
                           "deployment_id": response.json()['model_id'],
                           "endpoint": deployment_endpoint
                       }
                   )
               )
           except Exception as e:
               self.mlflow.log_param("deployment_error", str(e))
               return Event(
                   payload=PusherPayload(
                       model_uri="",
                       status="failed",
                       model_version=None
                   )
               )

Event-Based Data Flow
---------------------

Event and Payload System
~~~~~~~~~~~~~~~~~~~~~~~~~

Collie uses an event-driven architecture where components communicate through ``Event`` objects containing typed ``Payload`` data:

**Passing Data Between Components:**

1. **Standard Fields**: Use predefined payload fields for common data
2. **Extra Data**: Use ``extra_data`` for custom/experimental data

.. code-block:: python

   from collie import Event, TransformerPayload
   
   # Pass data using standard fields
   event = Event(
       payload=TransformerPayload(
           train_data=df,
           validation_data=val_df,
           test_data=test_df
       )
   )
   
   # Pass custom data using extra_data
   event = Event(
       payload=TransformerPayload(
           train_data=df,
           extra_data={
               "feature_names": ["age", "income"],
               "preprocessing_steps": ["scaling", "encoding"],
               "data_version": "v2.1"
           }
       )
   )
Accessing Data in Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Components receive data through ``event.payload`` and can access both standard fields and custom data:

.. code-block:: python

   from collie import Event, Trainer, TrainerPayload
   
   class MyTrainer(Trainer):
       def handle(self, event: Event) -> Event:
           # Access standard payload fields
           train_data = event.payload.train_data
           val_data = event.payload.validation_data
           
           # Access custom data from extra_data
           feature_names = event.payload.get_extra("feature_names", [])
           hyperparams = event.payload.get_extra("best_params", {})
           
           # Train model
           model = train_model(train_data, hyperparams)
           
           # Return new payload with results
           return Event(
               payload=TrainerPayload(
                   model=model,
                   extra_data={
                       "training_time": 120.5,
                       "n_epochs": 50
                   }
               )
           )

Data Passing Between Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Components pass data through Event payloads:

.. code-block:: python

   from collie import Event
   from collie.core import TransformerPayload, TrainerPayload, EvaluatorPayload

   # Transformer output
   class DataLoader(Transformer):
       def handle(self, event: Event) -> Event:
           # Load and process data
           X_train, X_test, y_train, y_test = load_and_split_data()
           
           return Event(
               payload=TransformerPayload(
                   train_data=pd.concat([X_train, y_train], axis=1),
                   validation_data=None,
                   test_data=pd.concat([X_test, y_test], axis=1)
               )
           )

   # Trainer uses transformer output
   class ModelTrainer(Trainer):
       def handle(self, event: Event) -> Event:
           train_data = event.payload.train_data
           X_train = train_data.drop("target", axis=1)
           y_train = train_data["target"]
           
           # ... train model ...
           
           return Event(
               payload=TrainerPayload(
                   model=trained_model
               )
           )

   # Evaluator uses trainer output
   class ModelEvaluator(Evaluator):
       def handle(self, event: Event) -> Event:
           model = event.payload.model
           test_data = event.payload.test_data
           X_test = test_data.drop("target", axis=1)
           y_test = test_data["target"]
           
           # Evaluate model
           y_pred = model.predict(X_test)
           accuracy = accuracy_score(y_test, y_pred)
           
           self.mlflow.log_metric("accuracy", accuracy)
           
           return Event(
               payload=EvaluatorPayload(
                   metrics=[{"accuracy": accuracy}],
                   is_better_than_production=True
               )
           )


Passing Custom Data
~~~~~~~~~~~~~~~~~~~

Each Payload has an ``extra_data`` field for custom data beyond standard fields:

**Standard Fields vs Extra Data:**

.. code-block:: python

   from collie import Event
   from collie.core import TransformerPayload, TrainerPayload
   
   # Use standard fields for common data
   class MyTransformer(Transformer):
       def handle(self, event: Event) -> Event:
           return Event(
               payload=TransformerPayload(
                   # Standard fields - strongly typed
                   train_data=train_df,
                   validation_data=val_df,
                   test_data=test_df,
                   
                   # Custom data - flexible
                   extra_data={
                       "feature_names": ["age", "income", "score"],
                       "data_source": "database",
                       "preprocessing_steps": ["normalization", "encoding"],
                       "data_version": "v2.1"
                   }
               )
           )

**Accessing Extra Data:**

.. code-block:: python

   class MyTrainer(Trainer):
       def handle(self, event: Event) -> Event:
           # Access standard fields
           train_data = event.payload.train_data
           
           # Access extra data using helper methods (recommended)
           feature_names = event.payload.get_extra("feature_names", [])
           data_version = event.payload.get_extra("data_version", "unknown")
           
           # Or check if exists first
           if event.payload.has_extra("preprocessing_steps"):
               steps = event.payload.get_extra("preprocessing_steps")
           
           # Log custom info
           self.mlflow.log_params({
               "data_version": data_version,
               "n_features": len(feature_names)
           })
           
           # Train model...
           
           return Event(
               payload=TrainerPayload(
                   model=model,
                   extra_data={
                       "training_time": training_time,
                       "early_stopping_epoch": best_epoch,
                       "optimizer": "Adam",
                       "train_loss": loss,  # Optional metrics via extra_data
                       "val_loss": val_loss,
                       **event.payload.extra_data  # Keep previous extra data
                   }
               )
           )

**Common Use Cases for Extra Data:**

1. **Transformer**: Feature engineering metadata, data quality metrics
2. **Tuner**: Trial history, convergence info, search space details
3. **Trainer**: Training curves, checkpoints, optimizer state
4. **Evaluator**: Detailed reports, plot file paths, per-class metrics
5. **Pusher**: Deployment endpoints, container IDs, rollback info

**Best Practices:**

- Use **standard fields** for data that all pipelines need (model, train_data, metrics)
- Use **extra_data** for pipeline-specific or experimental data
- Use **event.context** for metadata that doesn't belong in the payload (timestamps, versions)


Model Comparison
~~~~~~~~~~~~~~~~

Compare models using MLflow:

.. code-block:: python

   from collie import Event
   from collie.core import Evaluator, EvaluatorPayload
   from collie.core.enums.ml_models import MLflowModelStage, ModelFlavor
   

   class ModelEvaluator(Evaluator):
       def handle(self, event: Event) -> Event:
           # Load production model
           prod_model = self.load_latest_model(
               model_name=self.registered_model_name,
               stage=MLflowModelStage.PRODUCTION,
               flavor=ModelFlavor.SKLEARN
           )
           
           # Get new model and test data
           new_model = event.payload.model
           test_data = event.payload.test_data
           X_test = test_data.drop("target", axis=1)
           y_test = test_data["target"]
           
           new_score = new_model.score(X_test, y_test)
           
           if prod_model is not None:
               prod_score = prod_model.score(X_test, y_test)
               improvement = new_score - prod_score
               is_better = new_score > prod_score
           else:
               prod_score = 0
               improvement = new_score
               is_better = True
           
           self.mlflow.log_metrics({
               "new_model_score": new_score,
               "prod_model_score": prod_score,
               "improvement": improvement
           })
           
           return Event(
               payload=EvaluatorPayload(
                   metrics=[
                       {
                           "new_model_score": new_score,
                           "prod_model_score": prod_score,
                           "improvement": improvement
                       }
                   ],
                   is_better_than_production=is_better
               )
           )


Next Steps
----------

- Explore the :doc:`api/core` for detailed API reference
- See :doc:`mlflow_integration` for MLflow usage patterns
- Check out example pipelines in the ``examples/`` directory
