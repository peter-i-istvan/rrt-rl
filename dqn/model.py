import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.optim import Adam, SGD, Adagrad

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, hidden_activation='relu', out_activation=None, optimizer_type='adam', lr=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation

        self.model = self._create_model()
        self.optimizer = self._get_optimizer(optimizer_type, lr)

    def _get_optimizer(self, optimizer_type, lr):
        if optimizer_type == 'adam':
            return Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            return SGD(self.model.parameters(), lr=lr)
        elif optimizer_type == 'adagrad':
            return Adagrad(self.model.parameters(), lr=lr)
        else:
            print(f'{optimizer_type}, optimizer is not implemented')
            raise NotImplemented
    
    def _get_activation(self, activation_type):
        if activation_type == 'relu':
            return nn.ReLU()
        elif activation_type == 'tanh':
            return nn.Tanh()
        elif activation_type == 'sigmoid':
            return nn.Sigmoid()
        elif activation_type == 'softmax':
            return nn.Softmax(dim=-1)
        elif activation_type == 'leakyrelu':
            return nn.LeakyReLU()
        elif activation_type == None:
            return nn.Identity()
        else:
            print(f"No {self.activation_type} activation is implemented")
            raise NotImplementedError

    def _create_model(self):
        activation = self._get_activation(self.hidden_activation)

        layers = []
        
        start_dim = self.in_features

        for n in self.hidden_features:
            layers.extend([
                nn.Linear(start_dim, n),
                activation
            ])
            start_dim = n

        layers.extend([nn.Linear(start_dim, self.out_features), self._get_activation(self.out_activation)])

        model = Sequential(*layers)

        return model

    def forward(self, x):
        return self.model(x)