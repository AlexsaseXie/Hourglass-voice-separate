import sys
import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import configure, log_value
from torch.autograd.variable import Variable
from src.model.model import HourglassNet


mock_input = torch.randn(1, 512, 64)
mock_input = torch.unsqueeze(mock_input, 0)

print(mock_input.shape)

net = HourglassNet()

result, next_input = net(mock_input)

print(result.shape)
print(next_input.shape)
