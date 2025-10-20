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
   * ``self.mlflow.log_input_data()`` - Log input datasets
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
              
              # Log input data
              self.mlflow.log_input_data(
                  data=transformed_data,
                  context="training"
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

   The Evaluator component evaluates model performance.

   **MLflow Methods Available:**

   * ``self.mlflow.log_metrics()`` - Log evaluation metrics
   * ``self.mlflow.log_artifact()`` - Log evaluation plots
   * ``self.mlflow.log_dict()`` - Log detailed results
   * ``self.mlflow.load_model()`` - Load models for comparison

   **Example:**

   .. code-block:: python

      class MyEvaluator(Evaluator):
          def evaluate(self, context):
              model = context.data["model"]
              
              # Evaluate on test set
              metrics = calculate_metrics(model, test_data)
              
              # Log all metrics
              self.mlflow.log_metrics({
                  "accuracy": metrics["accuracy"],
                  "precision": metrics["precision"],
                  "recall": metrics["recall"],
                  "f1_score": metrics["f1"]
              })
              
              # Create and log confusion matrix
              plot_confusion_matrix(y_true, y_pred)
              self.mlflow.log_artifact("confusion_matrix.png")
              
              # Compare with production model
              prod_model = self.mlflow.load_model(
                  model_name="my_model",
                  stage="Production"
              )
              
              return {"should_promote": should_promote}

Pusher
~~~~~~

.. autoclass:: collie.core.pusher.pusher.Pusher
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   The Pusher component handles model deployment and registration.

   **MLflow Methods Available:**

   * ``self.mlflow.register_model()`` - Register model to registry
   * ``self.mlflow.set_tag()`` - Tag the deployed model
   * ``self.mlflow.log_param()`` - Log deployment parameters

   **Example:**

   .. code-block:: python

      class MyPusher(Pusher):
          def push(self, context):
              if not context.data.get("should_promote"):
                  return {"status": "skipped"}
              
              # Get current run
              run_id = self.mlflow.active_run().info.run_id
              model_uri = f"runs:/{run_id}/model"
              
              # Register model
              self.mlflow.register_model(
                  model_uri=model_uri,
                  model_name="my_model"
              )
              
              # Tag as deployed
              self.mlflow.set_tags({
                  "deployed": "true",
                  "deployment_date": datetime.now().isoformat()
              })
              
              return {"status": "success"}

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
