Collie MLOps Framework Documentation
=====================================

.. image:: https://img.shields.io/pypi/v/collie-mlops.svg
   :target: https://pypi.org/project/collie-mlops/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/collie-mlops.svg
   :target: https://pypi.org/project/collie-mlops/
   :alt: Python versions

Welcome to Collie
-----------------

Collie is a lightweight MLOps framework that provides a modular, event-driven architecture 
for building machine learning pipelines with deep MLflow integration.

**Key Features:**

* 🎯 **Modular Components**: Transform, Train, Tune, Evaluate, and Push
* 🔄 **Event-Driven**: Flexible workflow orchestration
* 📊 **MLflow Integration**: First-class MLflow support for tracking and model management
* 🚀 **Lightweight**: Simple setup, no complex dependencies
* 🔧 **Extensible**: Easy to add custom components

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install collie-mlops

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from collie.core import (
       Transformer,
       Trainer,
       Orchestrator,
       TransformerPayload,
       TrainerPayload
   )
   from collie import Event
   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier
   import pandas as pd

   # Define your Transformer
   class IrisTransformer(Transformer):
       def handle(self, event: Event) -> Event:
           data = load_iris()
           X = pd.DataFrame(data.data, columns=data.feature_names)
           y = pd.DataFrame(data.target, columns=["target"])
           train_data = pd.concat([X, y], axis=1)
           
           return Event(
               payload=TransformerPayload(
                   train_data=train_data,
                   validation_data=None,
                   test_data=None
               )
           )

   # Define your Trainer
   class IrisTrainer(Trainer):
       def handle(self, event: Event) -> Event:
           train_data = event.payload.train_data
           X = train_data.drop("target", axis=1)
           y = train_data["target"]
           
           model = RandomForestClassifier()
           model.fit(X, y)
           accuracy = model.score(X, y)
           
           # Log metrics to MLflow
           self.mlflow.log_metric("accuracy", accuracy)
           
           return Event(
               payload=TrainerPayload(
                   model=model,
                   train_loss=1 - accuracy,
                   val_loss=None
               )
           )

   # Run the pipeline
   orchestrator = Orchestrator(
       components=[IrisTransformer(), IrisTrainer()],
       tracking_uri="http://localhost:5000",
       experiment_name="iris_experiment",
       registered_model_name="iris_model"
   )
   orchestrator.run()

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   core_concepts
   mlflow_integration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/components
   api/contracts
   api/helpers

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/quick_start
   examples/sklearn_pipeline
   examples/pytorch_pipeline

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
