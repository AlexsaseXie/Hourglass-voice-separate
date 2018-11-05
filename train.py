import sys
import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import configure, log_value
from torch.autograd.variable import Variable
from src.model.model import HourglassNet, VoiceSeparateNet
from src.model.loss import J_loss
from src.generator.generator import Generator

config = {}
config['feature_size'] = [256, 128]
config['epochs'] = 1
config['batch_size'] = 1

#mock_input = torch.randn(3, 256, 128)
#mock_input = torch.unsqueeze(mock_input, 1)

net = VoiceSeparateNet(input_shape=config['feature_size'])

#result, next_input = net(mock_input)

gen = Generator(file_path='data/train/', feature_size=config['feature_size'])
train_data_iter = gen.get_file_data(batch_size = config['batch_size'])


for epoch in range(config['epochs']):
    train_loss = 0
    net.train()

    for batch_id in range(config['batch_size']):
        whole, left, right = next(train_data_iter)
        
        whole = np.reshape(whole, (config['batch_size'], 1, config['feature_size'][0], config['feature_size'][1]))
        left = np.reshape(left, (config['batch_size'], 1, config['feature_size'][0], config['feature_size'][1]))
        right = np.reshape(right, (config['batch_size'], 1, config['feature_size'][0], config['feature_size'][1]))  

        whole = Variable(torch.from_numpy(whole))

        mask1s, mask2s = net(whole)
        print(mask1s.shape, mask2s.shape)



         


