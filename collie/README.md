# Collie
This project is a component of the MLOps framework and is integrated with MLFlow 
for streamlined machine learning model management, including tracking, versioning, and deployment.  
It is expected to include the following components:  
* Transformer
* Tuner (Pytorch Tuner is not supported currently.)
* Trainer (Currently supports `XGBTrainer` and `PytorchTrainer`.)
* Evaluator
* Deployer (Not yet developed)


## TODO:
1. Use the MLFLOW to record the PytorchTrainer training metadata.(trainer)
2. Develope the PytorchTuner
3. Develope the Deployer.