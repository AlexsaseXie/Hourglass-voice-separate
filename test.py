import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from src.model.model import HourglassNet, VoiceSeparateNet
from src.model.loss import J_1track_loss, J_1track_whole_loss, J_2track_whole_loss, J_2track_loss
from src.generator.generator import Generator, TestGenerator
from src.utils import read_config
from src.utils import audio_transfer
from evaluation import bss_eval

if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config.yml")

# model name
model_name = config.model_path
net = VoiceSeparateNet(input_shape=config.feature_size)
net = nn.DataParallel(net)

if config.use_gpu:
    net.cuda()

# test data generator
gen = TestGenerator(file_path='data/test/', feature_size = config.feature_size)
test_data_iter = gen.get_file_data(batch_size = config.batch_size)

print(gen.whole_mel.shape)

# load pre-trained model
net.load_state_dict(torch.load(config.pretrain_modelpath))

# require grad
for param in net.parameters():
    param.requires_grad = True


test_data_size = gen.whole_mel.shape[0]

test_loss = 0


# estimation
estimation = {
    'GNSDR': np.zeros(2, dtype=np.float64),
    'GSIR': np.zeros(2, dtype=np.float64),
    'GSAR': np.zeros(2, dtype=np.float64)
}

total_length = 0

for batch_idx in range( test_data_size // config.batch_size):
    whole, left, right, phase, phase_acc, phase_voice = next(test_data_iter)
    
    whole = np.reshape(whole, (config.batch_size, 1, config.feature_size[0], config.feature_size[1]))
    left = np.reshape(left, (config.batch_size, 1, config.feature_size[0], config.feature_size[1]))
    right = np.reshape(right, (config.batch_size, 1, config.feature_size[0], config.feature_size[1]))  
    phase = np.reshape(phase, (config.batch_size, 1, config.feature_size[0], config.feature_size[1]))
    phase_acc = np.reshape(phase_acc, (config.batch_size, 1, config.feature_size[0], config.feature_size[1]))
    phase_voice = np.reshape(phase_voice, (config.batch_size, 1, config.feature_size[0], config.feature_size[1]))

    if config.use_gpu:
        whole = Variable(torch.from_numpy(whole)).cuda()
        left = Variable(torch.from_numpy(left)).cuda()
        right = Variable(torch.from_numpy(right)).cuda()
    else :
        whole = Variable(torch.from_numpy(whole))
        left = Variable(torch.from_numpy(left))
        right = Variable(torch.from_numpy(right))
        # phase = Variable(torch.from_numpy(phase))

    masks = net.predict(whole)

    # masks: batch * 2 * 512 * 64

    #loss = J_loss(maskss, whole , left, right)
    loss = J_2track_loss(masks, whole, left, right, use_gpu=config.use_gpu)
    print(loss)

    l = loss.data
    test_loss += l
    
    #log_value('train_loss_batch', l.cpu().numpy(), epoch * gen.whole_mel.shape[0] + batch_idx)

    print('test_loss_batch @ batch' + str(batch_idx) + ':' , l.cpu().numpy())


    # estimation
    masks = masks.data.cpu().numpy()
    left = left.data.cpu().numpy()
    whole = whole.data.cpu().numpy()
    right = right.data.cpu().numpy()
    # fix mask

    # matrix size: batch * 1(2) * 512 * 64
    batch_est, batch_length = bss_eval.estimate_batch(whole, left, right, masks, phase, phase_acc, phase_voice, config.batch_size)
    for k in estimation.keys():
        estimation[k] = estimation[k] + batch_est[k]

    total_length += batch_length

print('average_loss: ' + str(test_loss /  ( config.batch_size * (test_data_size // config.batch_size)) ))

# estimation
for k in estimation.keys():
    estimation[k] = estimation[k] / total_length
    print('%s on Accompaniment: [%lf], Voice: [%lf]' % (k, estimation[k][0].item(), estimation[k][1].item()))





         


