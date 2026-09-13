import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class HEAYieldStrengthGNN(nn.Module):
    """
    Graph Neural Network for HEA/MPEA Yield Strength Prediction.

    Nodes:
        Chemical elements

    Node features:
        6 elemental descriptors + composition fraction

    Edges:
        Element co-occurrence

    Graph-level features:
        Experimental conditions

    Output:
        Predicted yield strength
    """

    def __init__(
        self,
        node_features=7,
        hidden_dim=64,
        condition_dim=32,
        dropout=0.20
    ):
        super().__init__()

        self.dropout = dropout

        # Graph encoder
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Experimental condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, batch, conditions):

        # GCN layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # GCN layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # GCN layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Graph-level pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        graph_embedding = torch.cat(
            [x_mean, x_max],
            dim=1
        )

        # Experimental conditions
        condition_embedding = self.condition_encoder(
            conditions
        )

        # Combine graph + conditions
        combined = torch.cat(
            [graph_embedding, condition_embedding],
            dim=1
        )

        # Prediction
        output = self.predictor(combined)

        return output.squeeze(-1)


if __name__ == "__main__":

    print("=" * 60)
    print("HEA/MPEA GNN MODEL TEST")
    print("=" * 60)

    # Example graph with 3 elements
    x = torch.randn(3, 7)

    # Complete directed graph
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1]
        ],
        dtype=torch.long
    )

    # All nodes belong to graph 0
    batch = torch.zeros(
        3,
        dtype=torch.long
    )

    # Example 32-dimensional condition vector
    conditions = torch.randn(1, 32)

    # Create model
    model = HEAYieldStrengthGNN(
        node_features=7,
        hidden_dim=64,
        condition_dim=32
    )

    # Forward pass
    prediction = model(
        x=x,
        edge_index=edge_index,
        batch=batch,
        conditions=conditions
    )

    print("\nModel created successfully.")

    print("\nInput node shape:")
    print(x.shape)

    print("\nEdge shape:")
    print(edge_index.shape)

    print("\nPrediction:")
    print(prediction)

    print("\nPrediction shape:")
    print(prediction.shape)

    print("\nGNN model test completed successfully.")
    print("=" * 60)
