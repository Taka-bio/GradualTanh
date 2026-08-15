import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pennylane as qml

class HybridQNN(nn.Module):
    def __init__(self, input_size, hyper_params, norm, hidden_size=[512, 128]):
        super(HybridQNN, self).__init__()
        def unit(in_f, out_f):
            return [nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.SiLU()]
        def make_layer(layer_size):
            layer_list = unit(layer_size[0], layer_size[1])
            for i in range(len(layer_size)-3):
                layer_list += unit(layer_size[i+1], layer_size[i+2])
            layer_list += [nn.Linear(layer_size[-2], layer_size[-1])]
            return layer_list
        self.hyper_params = hyper_params
        self.num_encoding_params = self.hyper_params['num_encoding_layer'] * self.hyper_params['num_qubits']
        size = [input_size] + hidden_size + [self.num_encoding_params]
        self.encoder = nn.Sequential(
            *make_layer(size)
        )
        norm_dict = {
            "tanh": lambda x: F.tanh(x)*(np.pi/2),
            "gradual_tanh": lambda x: F.tanh(x/hyper_params["width"]) * (hyper_params["rotation"]),
            "layer_norm": nn.LayerNorm(self.num_encoding_params),
            "id": nn.Identity()
        }
        self.norm = norm_dict[norm]
        self.dev = qml.device('default.qubit', wires=self.hyper_params['num_qubits'])
        self.q_weights = nn.Parameter(torch.randn(self.num_encoding_params + self.hyper_params['reps'] * self.hyper_params['num_qubits']))
        self.quantum_net = qml.QNode(self._quantum_circuit, self.dev)
        self.hyper_params = hyper_params
        self.fc = nn.Linear(self.hyper_params['num_qubits'], 1)

    def _quantum_circuit(self, q_input, q_weights):
        for i in range(self.hyper_params['num_encoding_layer']):
            for j in range(self.hyper_params['num_qubits']):
                qml.RZ(q_input[i*self.hyper_params['num_qubits'] + j], wires=j)
                qml.RX(q_weights[i * self.hyper_params['num_qubits'] + j], wires=j)
            for j in range(self.hyper_params['num_qubits']):
                qml.CNOT(wires=[j, (j + 1) % self.hyper_params['num_qubits']])
        for i in range(self.hyper_params["reps"]):
            for j in range(self.hyper_params['num_qubits']):
                qml.RX(q_weights[self.num_encoding_params + i * self.hyper_params['num_qubits'] + j], wires=j)
            for j in range(self.hyper_params['num_qubits']):
                qml.CNOT(wires=[j, (j + 1) % self.hyper_params['num_qubits']])
        return [qml.expval(qml.PauliZ(j)) for j in range(self.hyper_params['num_qubits'])]

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = self.norm(x)
        x = torch.stack(self.quantum_net(x.T, self.q_weights)).T.reshape(batch_size, -1).float().to(x.device)
        x = self.fc(x)
        return x

class ClassicalNN(nn.Module):
    def __init__(self, input_size, hyper_params, hidden_size=[512, 128]):
        super(ClassicalNN, self).__init__()
        self.hyper_parameters = hyper_params
        def unit(in_f, out_f):
            return [nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.SiLU()]
        def make_layer(layer_size):
            layer_list = unit(layer_size[0], layer_size[1])
            for i in range(len(layer_size)-3):
                layer_list += unit(layer_size[i+1], layer_size[i+2])
            layer_list += [nn.Linear(layer_size[-2], layer_size[-1])]
            return layer_list
        self.num_encoding_params = hyper_params['num_encoding_layer'] * hyper_params['num_qubits']
        size = [input_size] + hidden_size + [self.num_encoding_params]
        self.encoder = nn.Sequential(
            *make_layer(size)
        )
        self.fc = nn.Linear(size[-1], 1)
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x