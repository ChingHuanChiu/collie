# Collie 🐕

[![PyPI version](https://badge.fury.io/py/collie-mlops.svg)](https://badge.fury.io/py/collie-mlops)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](docs/_build/html/index.html)

A Lightweight MLOps Framework for Machine Learning Workflows

📚 **[Full Documentation](docs/_build/html/index.html)** | 🚀 **[Getting Started](docs/_build/html/getting_started.html)** | 🔧 **[MLflow Integration Guide](docs/_build/html/mlflow_integration.html)** | 📖 **[API Reference](docs/_build/html/api/core.html)**

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
# Basic installation
pip install collie-mlops

# With specific ML framework support
pip install collie-mlops[sklearn]
pip install collie-mlops[pytorch]
pip install collie-mlops[xgboost]
pip install collie-mlops[lightgbm]
pip install collie-mlops[transformers]

# With all ML frameworks
pip install collie-mlops[all]
```

### Prerequisites

- Python >= 3.10
- MLflow tracking server (can be local or remote)

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

Collie supports multiple ML frameworks through its model flavor system:

| Framework | Status | Model I/O | Artifact Tracking |
|-----------|--------|-----------|-------------------|
| **PyTorch** | ✅ | ✅ | ✅ |
| **scikit-learn** | ✅ | ✅ | ✅ |
| **XGBoost** | ✅ | ✅ | ✅ |
| **LightGBM** | ✅ | ✅ | ✅ |
| **Transformers** | ✅ | ✅ | ✅ |


## Examples

Check out the [examples directory](./example/mlp) for complete working examples:
- [MLP Classification Pipeline](./example/mlp/mlp.ipynb) - End-to-end example with PyTorch

## Documentation

📚 **Complete documentation is available!**

- **[Getting Started Guide](docs/_build/html/getting_started.html)** - Your first Collie pipeline in 5 minutes
- **[MLflow Integration Guide](docs/_build/html/mlflow_integration.html)** - Complete guide to all `self.mlflow` methods
- **[Core Concepts](docs/_build/html/core_concepts.html)** - Understand Collie's architecture
- **[API Reference](docs/_build/html/api/core.html)** - Complete API documentation
- **[Quick Reference Card](docs/MLFLOW_QUICK_REFERENCE.md)** - Cheat sheet for MLflow methods

## Roadmap

- [x] Complete Sphinx documentation with MLflow methods
- [ ] CLI tool for project scaffolding
- [ ] TensorFlow/Keras support
- [ ] Distributed training support
- [ ] Model monitoring and drift detection
- [ ] Integration with Airflow/Kubeflow
- [ ] Published documentation on Read the Docs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Collie in your research, please cite:

```bibtex
@software{collie2025,
  author = {ChingHuanChiu},
  title = {Collie: A Lightweight MLOps Framework},
  year = {2025},
  url = {https://github.com/ChingHuanChiu/collie}
}
```
## Support

For questions and support, please open an issue on the [GitHub repository](https://github.com/ChingHuanChiu/collie/issues).


---

