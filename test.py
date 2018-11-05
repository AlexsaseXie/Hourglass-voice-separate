import sys
import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import configure, log_value
from torch.autograd.variable import Variable
from src.model.model import HourglassNet, VoiceSeparateNet
from src.model.loss import J_batch_loss, J_whole_loss, J_loss
from src.generator.generator import Generator
from src.utils import read_config

if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config.yml")

# model name
model_name = config.model_path
net = VoiceSeparateNet(input_shape=config.feature_size)

if config.use_gpu:
    net.cuda()

# test data generator
gen = Generator(file_path='data/test/', feature_size = config.feature_size)
test_data_iter = gen.get_file_data(batch_size = config.batch_size)

print(gen.whole_mel.shape)

# load pre-trained model
net.load_state_dict(torch.load(config.pretrain_modelpath))

# require grad
for param in net.parameters():
    param.requires_grad = True


test_data_size = gen.whole_mel.shape[0]

test_loss = 0

for batch_idx in range( test_data_size // config.batch_size):
    whole, left, right = next(test_data_iter)
    
    whole = np.reshape(whole, (config.batch_size, 1, config.feature_size[0], config.feature_size[1]))
    left = np.reshape(left, (config.batch_size, 1, config.feature_size[0], config.feature_size[1]))
    right = np.reshape(right, (config.batch_size, 1, config.feature_size[0], config.feature_size[1]))  

    if config.use_gpu:
        whole = Variable(torch.from_numpy(whole)).cuda()
        left = Variable(torch.from_numpy(left)).cuda()
        right = Variable(torch.from_numpy(right)).cuda()
    else :
        whole = Variable(torch.from_numpy(whole))
        left = Variable(torch.from_numpy(left))
        right = Variable(torch.from_numpy(right))

    masks = net.predict(whole)

    #loss = J_loss(maskss, whole , left, right)
    loss = J_batch_loss(masks, whole, left)
    print(loss)

    loss.backward()

    l = loss.data
    test_loss += l
    
    #log_value('train_loss_batch', l.cpu().numpy(), epoch * gen.whole_mel.shape[0] + batch_idx)

    print('test_loss_batch @ batch' + str(batch_idx) + ':' , l.cpu().numpy())

print('average_loss: ' + str(test_loss /  ( config.batch_size * (test_data_size // config.batch_size)) ))







         


