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
               payload=TrainerPayload(
                   model=model,
                   train_loss=loss,
                   val_loss=None
               )
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

**Purpose:** Model deployment

**Responsibilities:**
- Register model to MLflow Model Registry
- Transition model to appropriate stage
- Tag deployed models
- Handle deployment logic

**MLflow Usage:**
- Register model
- Set tags
- Transition model stages

**Example:**

.. code-block:: python

   from collie import Event
   from collie.core import PusherPayload
   
   class MyPusher(Pusher):
       def handle(self, event: Event) -> Event:
           if not event.payload.is_better_than_production:
               self.mlflow.log_param("deployment", "skipped")
               return Event(
                   payload=PusherPayload(
                       status="skipped",
                       model_version=None
                   )
               )
           
           # Get current run
           run_id = self.mlflow.active_run().info.run_id
           model_uri = f"runs:/{run_id}/model"
           
           # Register model
           model_version = self.mlflow.register_model(
               model_uri=model_uri,
               model_name=self.registered_model_name
           )
           
           # Tag the model
           self.mlflow.set_tags({
               "deployed": "true",
               "deployment_date": datetime.now().isoformat()
           })
           
           return Event(
               payload=PusherPayload(
                   status="deployed",
                   model_version=model_version.version
               )
           )

Event-Based Data Flow
---------------------

Event and Payload System
~~~~~~~~~~~~~~~~~~~~~~~~~

Collie uses an event-driven architecture where components communicate through ``Event`` objects containing typed ``Payload`` data:

.. code-block:: python

   from collie import Event
   from collie.contracts import PipelineContext
   
   # Event structure
   event = Event(
       type=EventType.DATA_READY,  # Optional event type
       payload=payload_object,      # Typed payload (TransformerPayload, TrainerPayload, etc.)
       context=PipelineContext()    # Shared context for metadata
   )

**Event Context for Metadata:**

The ``event.context`` can be used to store metadata that doesn't belong in the payload:

.. code-block:: python

   from collie import Event
   from collie.core import Transformer, TransformerPayload
   
   class MyComponent(Transformer):
       def handle(self, event: Event) -> Event:
           # Access shared context for metadata
           processing_start = time.time()
           
           # Do work...
           
           # Store metadata in context
           event.context.set("processing_time", time.time() - processing_start)
           event.context.set("component_version", "1.0.0")
           
           return Event(
               payload=TransformerPayload(
                   train_data=processed_data,
                   validation_data=None,
                   test_data=None
               ),
               context=event.context  # Pass context along
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
