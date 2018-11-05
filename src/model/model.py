import numpy as np
import torch
import torch.nn as nn
from torch.autograd.variable import Variable

class HourglassNet(nn.Module):
    def __init__(self,
                 input_shape=[512, 64],
                 ):
        """

        """
        super(HourglassNet, self).__init__()

        self.input_shape = input_shape

        # Encoder architecture
        self.n_conv1 = nn.Sequential (
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1))
        )

        self.n_conv2 = nn.Sequential (
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1))
        )

        self.n_conv3 = nn.Sequential (
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1))
        )

        self.n_conv4 = nn.Sequential (
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1))
        )

        self.additional_conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1))
        self.additional_conv2 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1))
        self.additional_conv3 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1))
        self.additional_conv4 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1))
        self.additional_conv5 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1))

        self.n_up1 = nn.Sequential ( 
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1)),
            nn.Upsample(size=[self.input_shape[0] // 8, self.input_shape[1] // 8])
        )

        self.n_up2 = nn.Sequential ( 
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1)),
            nn.Upsample(size=[self.input_shape[0] // 4, self.input_shape[1] // 4])
        )

        self.n_up3 = nn.Sequential ( 
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1)),
            nn.Upsample(size=[self.input_shape[0] // 2, self.input_shape[1] // 2])
        )

        self.n_up4 = nn.Sequential ( 
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1)),
            nn.Upsample(size=[self.input_shape[0], self.input_shape[1]])
        )

        self.mix_conv = nn.Sequential (
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 3,padding=(1,1)),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 1)
        )

        self.one_conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 1)
        self.one_conv2 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 1)
        self.one_conv3 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size = 1)


    def pred(self, freq_map):
        # print(freq_map.shape)
        n_1 = self.n_conv1(freq_map)
        n_2 = self.n_conv2(n_1)
        n_3 = self.n_conv3(n_2)

        n_4 = self.additional_conv4(n_3) + self.n_up1(self.additional_conv5(self.n_conv4(n_3)))
        n_5 = self.additional_conv3(n_2) + self.n_up2(n_4)
        n_6 = self.additional_conv2(n_1) + self.n_up3(n_5)
        n_7 = self.additional_conv1(freq_map) + self.n_up4(n_6)

        n_8 = self.mix_conv(n_7)
        
        output = self.one_conv2(n_8) 
        next_input = self.one_conv1(n_8) + self.one_conv3(output) + freq_map

        return output, next_input


    def forward(self, x):
        """
        Defines the forward pass for the network
        :param x: This will contain data based on the type of training that 
        you do.
        :return: outputs of the network, depending upon the architecture 
        """
        data = x
        # data : batch_size * 1(channel) * 512 * 64
        
        # batch_size = data.size()[1]
        
        outputs, next_inputs = self.pred(data)

        return outputs, next_inputs

    def test(self, data):
        """ Describes test behaviour of different models"""
        return 0

class VoiceSeparateNet(nn.Module):
    def __init__(self, 
            input_shape=[512, 64],
            pred_masks = 2
            ):

        super(VoiceSeparateNet, self).__init__()

        self.input_shape = input_shape
        self.pred_masks = 2

        self.hg1 = HourglassNet(input_shape=self.input_shape)
        self.hg2 = HourglassNet(input_shape=self.input_shape)

    def forward(self, x):

        mask1s, next_inputs = self.hg1(x)
        mask2s, _ = self.hg2(next_inputs)

        return mask1s, mask2s
