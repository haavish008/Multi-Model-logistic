def train_graph_neural_network(
    graph,
    epochs=100,
    learning_rate=0.01
):

    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv

    except ImportError:

        print(
            "PyTorch or PyTorch Geometric not installed. "
            "Skipping GNN module."
        )

        return None

    nodes = list(graph.nodes)

    node_index = {
        node: index
        for index, node in enumerate(nodes)
    }

    features = []
    labels = []

    for node in nodes:

        edges = [
            graph[node][neighbor]
            for neighbor in graph.neighbors(node)
        ]

        risks = [
            float(edge.get("risk_score", 0.0))
            for edge in edges
        ]

        slopes = [
            float(edge.get("slope_deg", 0.0))
            for edge in edges
            if np.isfinite(edge.get("slope_deg", np.nan))
        ]

        rainfall = [
            float(edge.get("rainfall_mm", 0.0))
            for edge in edges
            if np.isfinite(edge.get("rainfall_mm", np.nan))
        ]

        mean_risk = float(np.mean(risks)) if risks else 0.0
        mean_slope = float(np.mean(slopes)) if slopes else 0.0
        mean_rain = float(np.mean(rainfall)) if rainfall else 0.0

        features.append([
            node[0] / 1_000_000,
            node[1] / 1_000_000,
            mean_slope / 90,
            mean_rain / 100,
            mean_risk
        ])

        # Proxy label only; not an observed landslide label.
        labels.append(mean_risk)

    edge_index = []

    for u, v in graph.edges:

        edge_index.append([
            node_index[u],
            node_index[v]
        ])

        edge_index.append([
            node_index[v],
            node_index[u]
        ])

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(
            edge_index,
            dtype=torch.long
        ).t().contiguous(),
        y=torch.tensor(labels, dtype=torch.float32)
    )

    class LogisticsGCN(torch.nn.Module):

        def __init__(self):
            super().__init__()

            self.conv1 = GCNConv(5, 32)
            self.conv2 = GCNConv(32, 16)
            self.output = torch.nn.Linear(16, 1)

        def forward(self, data):

            x = F.relu(
                self.conv1(data.x, data.edge_index)
            )

            x = F.dropout(
                x,
                p=0.2,
                training=self.training
            )

            x = F.relu(
                self.conv2(x, data.edge_index)
            )

            return torch.sigmoid(
                self.output(x).squeeze(-1)
            )

    model = LogisticsGCN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        predictions = model(data)

        loss = F.mse_loss(
            predictions,
            data.y
        )

        loss.backward()
        optimizer.step()

        history.append({
            "epoch": epoch + 1,
            "loss": float(loss.item())
        })

    model.eval()

    with torch.no_grad():
        predictions = model(data).cpu().numpy()

    return model, pd.DataFrame(history), predictions
