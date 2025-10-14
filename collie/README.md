# Collie 🐕

A Lightweight MLOps Framework for Machine Learning Workflows

## Overview

Collie is a modern MLOps framework designed to streamline machine learning workflows by providing a component-based architecture integrated with MLflow. It enables data scientists and ML engineers to build, deploy, and manage ML pipelines with ease through modular components that handle different stages of the ML lifecycle.

## Features

- **Component-Based Architecture**: Modular design with specialized components for each ML workflow stage
- **MLflow Integration**: Built-in experiment tracking, model registration, and deployment capabilities
- **Pipeline Orchestration**: Seamless workflow management with event-driven architecture
- **Model Management**: Automated model versioning, staging, and promotion
- **Framework Agnostic**: Supports multiple ML frameworks (PyTorch, scikit-learn, XGBoost, LightGBM, Transformers)

## Architecture

Collie follows an event-driven architecture with the following core components:

- **Transformer**: Data preprocessing and feature engineering
- **Tuner**: Hyperparameter optimization
- **Trainer**: Model training and validation
- **Evaluator**: Model evaluation and comparison
- **Pusher**: Model deployment and registration
- **Orchestrator**: Workflow coordination and execution

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

Here's a simple example of building an ML pipeline with Collie:

```python
from collie.core import (
    Transformer, Trainer, Evaluator, Pusher, Orchestrator
)
from collie import Event

# Define your components
class MyTransformer(Transformer):
    def handle(self, event) -> Event:
        # Your data transformation logic
        pass

class MyTrainer(Trainer):
    def handle(self, event) -> Event:
        # Your model training logic
        pass

class MyEvaluator(Evaluator):
    def handle(self, event) -> Event:
        # Your model evaluation logic
        pass

class MyPusher(Pusher):
    def handle(self, event) -> Event:
        # Your model deployment logic
        pass

# Create and run the pipeline
orchestrator = Orchestrator(
    tracking_uri="http://localhost:5000",
    components=[
        MyTransformer(),
        MyTrainer(),
        MyEvaluator(),
        MyPusher()
    ],
    experiment_name="MyExperiment",
    registered_model_name="MyModel"
)

orchestrator.run()
```

## Components

### Transformer
Handles data preprocessing, feature engineering, and data validation.

```python
class CustomTransformer(Transformer):
    def handle(self, event) -> Event:
        # Process your data
        processed_data = self.preprocess(raw_data)
        return Event(payload=TransformerPayload(train_data=processed_data))
```

### Tuner
Performs hyperparameter optimization using various strategies.

```python
class CustomTuner(Tuner):
    def handle(self, event) -> Event:
        # Optimize hyperparameters
        best_params = self.optimize(search_space)
        return Event(payload=TunerPayload(hyperparameters=best_params))
```

### Trainer
Trains machine learning models with automatic experiment tracking.

```python
class CustomTrainer(Trainer):
    def handle(self, event) -> Event:
        # Train your model
        model = self.train(data, hyperparameters)
        return Event(payload=TrainerPayload(model=model))
```

### Evaluator
Evaluates model performance and decides on deployment.

```python
class CustomEvaluator(Evaluator):
    def handle(self, event) -> Event:
        # Evaluate model performance
        metrics = self.evaluate(model, test_data)
        is_better = self.compare_with_production(metrics)
        return Event(payload=EvaluatorPayload(
            metrics=metrics, 
            is_better_than_production=is_better
        ))
```

### Pusher
Handles model deployment and registration.

```python
class CustomPusher(Pusher):
    def handle(self, event) -> Event:
        # Deploy model to production
        model_uri = self.deploy(model)
        return Event(payload=PusherPayload(model_uri=model_uri))
```

## Configuration

### MLflow Setup

Start MLflow tracking server:

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

## Supported Frameworks

Collie supports multiple ML frameworks through its model flavor system: the currently supported flavors are

- **PyTorch**: Deep learning models
- **scikit-learn**: Traditional ML algorithms
- **XGBoost**: Gradient boosting
- **LightGBM**: Gradient boosting
- **Transformers**: transformers models


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue on the GitHub repository.

---

