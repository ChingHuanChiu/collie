MLflow Integration
==================

Collie provides deep integration with MLflow for experiment tracking, model logging, 
and model management. This guide explains how to use MLflow features within Collie components.

Overview
--------

Every Collie component (Transformer, Trainer, Tuner, Evaluator, Pusher) has access to 
MLflow functionality through the ``self.mlflow`` attribute. This provides a clean interface 
to all MLflow operations.

MLflow Interface
----------------

The MLflow Integration Contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collie components inherit from ``MLflowIntegration`` which provides:

.. autoclass:: collie.contracts.mlflow.MLflowIntegration
   :members:
   :undoc-members:
   :show-inheritance:

Available MLflow Methods
------------------------

Core Logging Methods
~~~~~~~~~~~~~~~~~~~~

Log Parameters
^^^^^^^^^^^^^^

Log parameters to track configuration and hyperparameters:

.. code-block:: python

   class MyTrainer(Trainer):
       def train(self, context):
           # Log single parameter
           self.mlflow.log_param("learning_rate", 0.01)
           
           # Log multiple parameters
           self.mlflow.log_params({
               "batch_size": 32,
               "epochs": 100,
               "optimizer": "adam"
           })

Log Metrics
^^^^^^^^^^^

Log metrics to track model performance:

.. code-block:: python

   class MyEvaluator(Evaluator):
       def evaluate(self, context):
           # Log single metric
           self.mlflow.log_metric("accuracy", 0.95)
           
           # Log multiple metrics
           self.mlflow.log_metrics({
               "precision": 0.93,
               "recall": 0.94,
               "f1_score": 0.935
           })
           
           # Log metric at specific step (for training curves)
           for epoch in range(100):
               loss = train_one_epoch()
               self.mlflow.log_metric("loss", loss, step=epoch)

Log Artifacts
^^^^^^^^^^^^^

Log files, plots, and other artifacts:

.. code-block:: python

   class MyEvaluator(Evaluator):
       def evaluate(self, context):
           # Log a single file
           self.mlflow.log_artifact("confusion_matrix.png")
           
           # Log entire directory
           self.mlflow.log_artifacts("evaluation_results/")
           
           # Log dict as JSON artifact
           self.mlflow.log_dict({
               "feature_importance": importance_dict
           }, "feature_importance.json")

Load Models
^^^^^^^^^^^

Load models from MLflow registry:

.. code-block:: python

   class MyEvaluator(Evaluator):
       def evaluate(self, context):
           # Load production model
           prod_model = self.mlflow.load_model(
               model_name="iris_classifier",
               stage="Production"
           )
           
           # Load specific version
           latest_model = self.mlflow.load_model(
               model_name="iris_classifier", 
               version=5
           )



Advanced Features
-----------------

Run Management
~~~~~~~~~~~~~~

Context Managers
^^^^^^^^^^^^^^^^

Use context managers for nested runs:

.. code-block:: python

   class MyTrainer(Trainer):
       def train(self, context):
           # Parent run (handled by Orchestrator)
           with self.mlflow.start_run(run_name="parent_run"):
               
               # Child run for cross-validation
               for fold in range(5):
                   with self.mlflow.start_run(
                       run_name=f"fold_{fold}",
                       nested=True
                   ):
                       self.mlflow.log_metric(f"cv_score_fold_{fold}", score)

Experiment Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get current experiment
   experiment = self.mlflow.get_experiment()
   
   # Get experiment by name
   experiment = self.mlflow.get_experiment_by_name("my_experiment")
   
   # Set experiment
   self.mlflow.set_experiment("new_experiment")

Tagging
~~~~~~~

Add tags for better organization:

.. code-block:: python

   class MyTrainer(Trainer):
       def train(self, context):
           # Set tags
           self.mlflow.set_tag("model_type", "random_forest")
           self.mlflow.set_tags({
               "team": "ml-team",
               "project": "customer-churn",
               "version": "v2"
           })



MLflow Methods Reference
------------------------

Complete list of available MLflow methods through ``self.mlflow``:


For more details on MLflow functionality, see the 
`MLflow documentation <https://mlflow.org/docs/latest/>`_.
