Core API Reference
==================

This section documents the core Collie API including all main components.

Core Components
---------------

Transformer
~~~~~~~~~~~

.. autoclass:: collie.core.transform.transform.Transformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   The Transformer component is responsible for data preprocessing and feature engineering.

   **MLflow Methods Available:**

   * ``self.mlflow.log_param()`` - Log transformation parameters
   * ``self.mlflow.log_pd_data()`` - Log pandas DataFrame datasets
   * ``self.mlflow.log_artifact()`` - Log transformation artifacts
   * ``self.mlflow.log_dict()`` - Log transformation statistics

   **Example:**

   .. code-block:: python

      class MyTransformer(Transformer):
          def transform(self, context):
              # Log transformation parameters
              self.mlflow.log_params({
                  "scaling_method": "standard",
                  "n_features": 10
              })
              
              # Process data
              transformed_data = preprocess(data)
              
              # Log pandas DataFrame
              self.mlflow.log_pd_data(
                  data=transformed_data,
                  context="training",
                  source="preprocessing"
              )
              
              return {"data": transformed_data}

Trainer
~~~~~~~

.. autoclass:: collie.core.trainer.trainer.Trainer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   The Trainer component handles model training.

   **MLflow Methods Available:**

   * ``self.mlflow.log_param()`` - Log hyperparameters
   * ``self.mlflow.log_metric()`` - Log training metrics
   * ``self.mlflow.log_model()`` - Log the trained model
   * ``self.mlflow.log_artifact()`` - Log training artifacts

   **Example:**

   .. code-block:: python

      class MyTrainer(Trainer):
          def train(self, context):
              # Log hyperparameters
              self.mlflow.log_params({
                  "learning_rate": 0.01,
                  "batch_size": 32,
                  "epochs": 100
              })
              
              # Train model with metric logging
              for epoch in range(100):
                  loss = train_one_epoch()
                  self.mlflow.log_metric("loss", loss, step=epoch)
              
              # Model is automatically logged by Collie
              return {"model": model}

Tuner
~~~~~

.. autoclass:: collie.core.tuner.tuner.Tuner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   The Tuner component performs hyperparameter optimization.

   **MLflow Methods Available:**

   * ``self.mlflow.log_params()`` - Log tuned hyperparameters
   * ``self.mlflow.log_metrics()`` - Log tuning metrics
   * ``self.mlflow.log_dict()`` - Log tuning results
   * ``self.mlflow.start_run()`` - Create nested runs for each trial

   **Example:**

   .. code-block:: python

      class MyTuner(Tuner):
          def tune(self, context):
              best_params = {}
              best_score = 0
              
              # Try different hyperparameters
              for params in param_grid:
                  with self.mlflow.start_run(nested=True):
                      # Log trial parameters
                      self.mlflow.log_params(params)
                      
                      # Train and evaluate
                      score = train_and_evaluate(params)
                      self.mlflow.log_metric("cv_score", score)
                      
                      if score > best_score:
                          best_score = score
                          best_params = params
              
              # Log best parameters
              self.mlflow.log_dict(best_params, "best_params.json")
              
              return {"best_params": best_params}

Evaluator
~~~~~~~~~

.. autoclass:: collie.core.evaluator.evaluator.Evaluator
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   The Evaluator component evaluates model performance and determines if the model
   passes the evaluation criteria. It does NOT register models or transition stages.

   **MLflow Methods Available:**

   * ``self.mlflow.log_metrics()`` - Log evaluation metrics
   * ``self.mlflow.log_artifact()`` - Log evaluation plots and reports
   * ``self.mlflow.log_dict()`` - Log detailed evaluation results
   * ``self.mlflow.load_model()`` - Load models for comparison

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
              
              # Evaluate on test set
              y_pred = model.predict(X_test)
              accuracy = calculate_accuracy(y_test, y_pred)
              
              # Log evaluation metrics
              self.mlflow.log_metrics({
                  "test_accuracy": accuracy,
                  "test_samples": len(y_test)
              })
              
              # Create and log confusion matrix plot
              plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")
              self.mlflow.log_artifact("confusion_matrix.png")
              
              # Optional: Compare with production model
              try:
                  prod_model = self.load_latest_model(
                      model_name=self.registered_model_name,
                      stage=MLflowModelStage.PRODUCTION,
                      flavor=ModelFlavor.SKLEARN
                  )
                  prod_pred = prod_model.predict(X_test)
                  prod_accuracy = calculate_accuracy(y_test, prod_pred)
                  
                  self.mlflow.log_metric("prod_accuracy", prod_accuracy)
                  is_better = accuracy > prod_accuracy
              except Exception:
                  # No production model exists yet
                  is_better = accuracy > 0.85  # Use threshold
              
              # Return evaluation results
              # The Pusher component will handle model registration
              payload = EvaluatorPayload(
                  metrics={"accuracy": accuracy},
                  pass_evaluation=is_better  # This flag tells Pusher to proceed
              )
              return Event(payload=payload)

Pusher
~~~~~~

.. autoclass:: collie.core.pusher.pusher.Pusher
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   The Pusher component handles model registration to MLflow Model Registry and 
   stage transitions. It automatically registers models and transitions them to 
   the configured target stage (e.g., Staging, Production).

   **Configuration:**

   * ``registered_model_name`` - The name for the model in MLflow Registry
   * ``target_stage`` - The MLflow stage to transition to (default: STAGING)

   **MLflow Methods Available:**

   * ``self.mlflow.register_model()`` - Register model to registry
   * ``self.mlflow.transition_model_version_stage()`` - Transition model stage
   * ``self.mlflow.set_registered_model_tag()`` - Tag the registered model
   * ``self.mlflow.set_model_version_tag()`` - Tag specific model version
   * ``self.mlflow.log_param()`` - Log deployment parameters

   **Example:**

   .. code-block:: python

      from collie import Event
      from collie.core import PusherPayload
      from collie.core.enums.ml_models import MLflowModelStage
      
      # Configure Pusher with target stage
      pusher = MyPusher(
          registered_model_name="my_model",
          target_stage=MLflowModelStage.PRODUCTION  # Auto-promote to Production
      )
      
      class MyPusher(Pusher):
          def handle(self, event: Event) -> Event:
              # Check if evaluation passed
              if not event.payload.pass_evaluation:
                  self.mlflow.log_param("registration_status", "skipped")
                  payload = PusherPayload(
                      model_uri="",
                      registered=False
                  )
                  return Event(payload=payload)
              
              # Model registration and stage transition is automatic
              # The base Pusher class handles:
              # 1. Registering the model
              # 2. Transitioning to target_stage
              # 3. Setting appropriate tags
              
              # You can add custom logic here
              self.mlflow.log_param("registration_status", "success")
              
              # The model is already registered by base class
              # Return success payload
              payload = PusherPayload(
                  model_uri=f"models:/{self.registered_model_name}/{self.target_stage}",
                  registered=True
              )
              return Event(payload=payload)
              
              return Event(
                  payload=PusherPayload(
                      model_uri=model_uri,
                      status="deployed",
                      model_version=str(model_version.version)
                  )
              )

Orchestrator
~~~~~~~~~~~~

.. autoclass:: collie.core.orchestrator.orchestrator.Orchestrator
   :members:
   :undoc-members:
   :show-inheritance:

   The Orchestrator coordinates the execution of all components in the pipeline.

   **Example:**

   .. code-block:: python

      orchestrator = Orchestrator(
          components=[
              MyTransformer(),
              MyTrainer(),
              MyEvaluator(),
              MyPusher()
          ],
          tracking_uri="http://localhost:5000",
          registered_model_name="my_model",
          experiment_name="my_experiment"
      )
      
      # Run the entire pipeline
      orchestrator.run()

Core Models
-----------

Data Payload Classes
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: collie.core.models.TransformerPayload
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: collie.core.models.TrainerPayload
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: collie.core.models.TunerPayload
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: collie.core.models.EvaluatorPayload
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: collie.core.models.PusherPayload
   :members:
   :undoc-members:
   :show-inheritance:

Enumerations
~~~~~~~~~~~~

.. autoclass:: collie.core.enums.ml_models.ModelFlavor
   :members:
   :undoc-members:
   :show-inheritance:

   Supported model flavors for MLflow logging.

.. autoclass:: collie.core.enums.ml_models.MLflowModelStage
   :members:
   :undoc-members:
   :show-inheritance:

   MLflow Model Registry stages.

.. autoclass:: collie.core.enums.components.ComponentType
   :members:
   :undoc-members:
   :show-inheritance:

   Component types in the pipeline.
