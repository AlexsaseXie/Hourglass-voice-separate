import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from src.model.model_with_relu import HourglassNet, VoiceSeparateNet
from src.model.loss import J_1track_loss, J_1track_whole_loss, J_2track_whole_loss, J_2track_loss
from src.generator.generator import Generator
from src.utils import read_config

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

# train data generator
gen = Generator(file_path='data/train/', feature_size=config.feature_size)
train_data_iter = gen.get_file_data(batch_size = config.batch_size)

# validate data generator
gen_v = Generator(file_path='data/validate/', feature_size=config.feature_size)
validate_data_iter = gen_v.get_file_data(batch_size = config.batch_size)

# load pre-trained model
if config.preload_model:
    net.load_state_dict(torch.load(config.pretrain_modelpath))

# require grad
for param in net.parameters():
    param.requires_grad = True


# optimizer
if config.optim == "sgd":
    optimizer = optim.SGD(
        [para for para in net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        momentum=0.9, lr=config.lr, nesterov=False)

elif config.optim == "adam":
    optimizer = optim.Adam(
        [para for para in net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay, lr=config.lr)


train_data_size = gen.whole_mel.shape[0]
validate_data_size = gen_v.whole_mel.shape[0]

for epoch in range(config.epochs):
    train_loss = 0
    net.train()

    for batch_idx in range( train_data_size // config.batch_size):
        optimizer.zero_grad()

        whole, left, right = next(train_data_iter)
        
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

        maskss = net(whole)

        #loss = J_1track_whole_loss(maskss, whole, left, use_gpu=config.use_gpu)

        #if (epoch <= config.epochs // 5):
        loss = J_2track_whole_loss(maskss, whole, left, right,use_gpu=config.use_gpu)
        #else:
        #    loss = J_2track_loss(maskss[-1], whole, left, right, use_gpu=config.use_gpu)

        loss.backward()

        # Clip the gradient to fixed value to stabilize training.
        
        torch.nn.utils.clip_grad_norm(net.parameters(), 20)
        optimizer.step()
        
        l = loss.data
        train_loss += l
        
    print('finish epoch ' + str(epoch) + ' :' + str(train_loss /  (config.batch_size * ( train_data_size //config.batch_size) )) )

    if (epoch == int(config.epochs * 0.8)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.lr / 5

    # validate
    validate_loss = 0
    for batch_idx in range( validate_data_size // config.batch_size):
        whole, left, right = next(validate_data_iter)
        
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

        maskss = net(whole)

        loss = J_2track_whole_loss(maskss, whole, left, right,use_gpu=config.use_gpu)
        
        l = loss.data
        validate_loss += l

    print('finish epoch ' + str(epoch) + ' validate:' + str(validate_loss /  (config.batch_size * ( validate_data_size //config.batch_size) )) )
    torch.save(net.state_dict(), "trained_models/{}.pth".format(model_name))


    






         


