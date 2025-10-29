# Collie 🐕

[![PyPI version](https://badge.fury.io/py/collie-mlops.svg)](https://badge.fury.io/py/collie-mlops)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](docs/_build/html/index.html)
[![codecov](https://codecov.io/gh/ChingHuanChiu/collie/branch/main/graph/badge.svg)](https://codecov.io/gh/ChingHuanChiu/collie)

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

Collie supports multiple ML frameworks through its model flavor system currently:

-  **PyTorch** 
-  **scikit-learn**
-  **XGBoost** 
-  **LightGBM**
-  **Transformers**


## Documentation

[Here you are]( https://collie-mlops.readthedocs.io/en/latest/getting_started.html )

## Roadmap

- [ ] TensorFlow/Keras support
- [ ] Model monitoring and drift detection
- [ ] Integration with Airflow/Kubeflow
- [ ] Automatically start the MLflow service

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

---

