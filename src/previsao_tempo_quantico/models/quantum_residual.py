from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


class QuantumResidualRegressor(nn.Module):
    """Corretor residual hibrido: base classica + camada quantica variacional."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_qubits: int = 6,
        n_layers: int = 3,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()

        if n_qubits < 2:
            raise ValueError("n_qubits deve ser >= 2")

        self.n_qubits = n_qubits

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh(),
        )

        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def qnode(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.residual_head = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        q_features = self.quantum_layer(encoded)
        residual = self.residual_head(q_features)
        return baseline + residual
