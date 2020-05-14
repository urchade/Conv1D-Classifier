"""model.py
Author: Urchade Zaratiana
"""
import torch.nn as nn


class Con1dClassifier(nn.Module):
    def __init__(self, num_layers, in_channels, num_kernels, kernel_size, n_outputs, act=nn.ReLU()):
        super().__init__()

        self.layers = nn.ModuleList()

        # hidden layers
        for i in range(num_layers):
            conv_layer = nn.Conv1d(in_channels, num_kernels, kernel_size, padding=int(kernel_size / 2))
            norm = nn.BatchNorm1d(num_kernels)
            layer = [conv_layer, act, norm]
            self.layers.append(nn.Sequential(*layer))
            in_channels = num_kernels

        # output layer
        self.layers.append(nn.Conv1d(num_kernels, n_outputs, kernel_size, padding=int(kernel_size / 2)))

    def forward(self, x):

        for i, layer in enumerate(self.layers):

            condition = i > 1 + i < len(self.layers)

            if condition:
                x = layer(x) + x  # Skip connection
            else:
                x = layer(x)

        return x.mean(dim=-1)  # Global Average Pooling
