import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from collie.core import (
    Transformer,
    Tuner,
    Trainer,
    Evaluator,
    Pusher,
    Event,
    TrainerPayload,
    TransformerPayload,
    EvaluatorPayload,
    PusherPayload,
    AirflowOrchestrator
)


num_samples = 1000
input_dim = 20   
num_classes = 4


class MLPTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def handle(self, event) -> Event:

        X = torch.randn(num_samples, input_dim)
        y = torch.randint(0, num_classes, (num_samples,))

        X_data = pd.DataFrame(X.numpy(), columns=[f"feature_{i}" for i in range(input_dim)])
        y_data = pd.DataFrame(y.numpy(), columns=["label"])

        train_data = pd.concat([X_data, y_data], axis=1)

        return Event(
            payload=TransformerPayload(
                train_data=train_data,
                validation_data=None,
                test_data=None
            )
        )

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    
class MLPTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.model = SimpleClassifier()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def handle(self, event):
        
        train_data = event.payload.train_data
        X = train_data.drop("label", axis=1)
        y = train_data["label"]

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        epochs = 10
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.log_metric("learning rate", self.scheduler.get_last_lr()[0], step=epoch)
            self.log_metric("loss", round(total_loss/len(dataloader), 3), step=epoch)
            
        return Event(
            payload=TrainerPayload(
                model=self.model,
                train_loss=total_loss/len(dataloader),
                val_loss=None
            )
        )


class MLPEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    def handle(self, event):
        model = event.payload.model
        train_loss = event.payload.train_loss
        #mock the production metrics
        production_metric = 10

        return Event(
            payload=EvaluatorPayload(
                metrics={"Experiment": train_loss, "Production": production_metric},
                greater_is_better=False
            )
        )


class MLPPusher(Pusher):
    def __init__(self):
        super().__init__()

    def handle(self, event):
        return Event(
            payload=PusherPayload(
                model_uri="mlp_model_uri",
            )
        )
    

if __name__ == "__main__":

    orchestrator = AirflowOrchestrator(
        dag_id="mlp_pipeline",
        tracking_uri="mysql://root:password@localhost:3306/collie",
        components=[
            MLPTransformer(),
            MLPTrainer(),
            MLPEvaluator(),
            MLPPusher()
        ],
        mlflow_tags={"Example": "MLP"},
        experiment_name="MLP",
    )
    orchestrator.run()