import sys
import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import configure, log_value
from torch.autograd.variable import Variable
from src.model.model import HourglassNet
from src.generator.generator import Generator

mock_input = torch.randn(1, 512, 64)
mock_input = torch.unsqueeze(mock_input, 0)

print(mock_input.shape)

net = HourglassNet()

result, next_input = net(mock_input)


gen = Generator(file_path='data/train/')
print(gen.files)
train_data_iter = gen.get_file_data(batch_size = 1)

a, b, c= next(train_data_iter)
