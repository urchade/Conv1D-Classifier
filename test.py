import torch.nn as nn
import torch
from model import Con1dClassifier

in_channels = 20


def sample_generator():
    """Function returning a generator to generate random samples"""
    while True:
        input_tensor = torch.randn(size=(20, in_channels, 20), dtype=torch.float32).reshape(20, in_channels, 20)
        target_tensor = torch.FloatTensor(input_tensor[:, 0, :])
        yield input_tensor, target_tensor


nets = Con1dClassifier(num_layers=100, in_channels=in_channels, num_kernels=16, kernel_size=3, n_outputs=10)

for a, b in sample_generator():
    a = nets.forward(a)

    print(a)
    break