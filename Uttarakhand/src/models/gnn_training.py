
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def prepare_gnn_data(
    graph,
    node_features,
    node_labels,
    train_mask,
    test_mask
):

    nodes = list(graph.nodes())
    node_to_index = {
        node: index
        for index, node in enumerate(nodes)
    }

    x = np.asarray(node_features, dtype=np.float32)
    y = np.asarray(node_labels, dtype=np.int64)

    if len(x) != len(nodes):
        raise ValueError(
            "Node feature count must match graph node count."
        )

    if len(y) != len(nodes):
        raise ValueError(
            "Node label count must match graph node count."
        )

    if x.ndim != 2:
        raise ValueError("Node features must be a 2D matrix.")

    edges = []

    for u, v in graph.edges():
        ui = node_to_index[u]
        vi = node_to_index[v]

        edges.append([ui, vi])
        edges.append([vi, ui])

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).T.contiguous()

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(
            train_mask,
            dtype=torch.bool
        ),
        test_mask=torch.tensor(
            test_mask,
            dtype=torch.bool
        )
    )

    if len(data.train_mask) != len(nodes):
        raise ValueError("Invalid training mask.")

    if len(data.test_mask) != len(nodes):
        raise ValueError("Invalid testing mask.")

    if not data.train_mask.any():
        raise ValueError("Training mask is empty.")

    if not data.test_mask.any():
        raise ValueError("Testing mask is empty.")

    if (data.train_mask & data.test_mask).any():
        raise ValueError(
            "Training and testing masks must not overlap."
        )

    return data, node_to_index


class LogisticsGCN(torch.nn.Module):

    def __init__(
        self,
        input_features,
        hidden_features=32,
        output_classes=2
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_features,
            hidden_features
        )

        self.conv2 = GCNConv(
            hidden_features,
            hidden_features
        )

        self.output = torch.nn.Linear(
            hidden_features,
            output_classes
        )

    def forward(self, data):

        x = self.conv1(
            data.x,
            data.edge_index
        )

        x = F.relu(x)
        x = F.dropout(
            x,
            p=0.3,
            training=self.training
        )

        x = self.conv2(
            x,
            data.edge_index
        )

        x = F.relu(x)

        return self.output(x)


def train_gnn(
    data,
    model_path,
    epochs=100,
    learning_rate=0.001
):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    data = data.to(device)

    model = LogisticsGCN(
        input_features=data.x.shape[1],
        output_classes=int(data.y.max().item()) + 1
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0001
    )

    history = []

    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()

        output = model(data)

        loss = F.cross_entropy(
            output[data.train_mask],
            data.y[data.train_mask]
        )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            predictions = model(data).argmax(dim=1)

            train_accuracy = (
                predictions[data.train_mask]
                == data.y[data.train_mask]
            ).float().mean().item()

            test_accuracy = (
                predictions[data.test_mask]
                == data.y[data.test_mask]
            ).float().mean().item()

        history.append({
            "epoch": epoch + 1,
            "loss": float(loss.item()),
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        })

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Test accuracy: {test_accuracy:.4f}"
            )

    model_path = Path(model_path)
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": data.x.shape[1],
            "output_classes": int(data.y.max().item()) + 1
        },
        model_path
    )

    model.eval()

    with torch.no_grad():
        predictions = (
            model(data)[data.test_mask]
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        actual = (
            data.y[data.test_mask]
            .cpu()
            .numpy()
        )

    metrics = {
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(
            actual,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            actual,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            actual,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    return model, history, metrics


def save_gnn_history(history, output_path):
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(history).to_csv(
        output_path,
        index=False
    )
