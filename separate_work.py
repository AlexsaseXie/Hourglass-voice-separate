import sys
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd.variable import Variable
from src.model.model_with_relu import HourglassNet, VoiceSeparateNet
from src.utils import read_config
import librosa
from src.utils import audio_transfer

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


# load pre-trained model
net.load_state_dict(torch.load(config.pretrain_modelpath))

net.eval()

def windows(data, window_size, stride):
    """
    data : H * W, window on W 
    """
    start = 0
    while start < data.shape[1]:
        yield start, start + window_size
        start += stride


file_name = input('file name:')

audio_clip, sr = librosa.load(file_name, sr=None, mono=True)
whole_clip, _ = librosa.core.spectrum._spectrogram(audio_clip, n_fft=config.feature_size[0] * 2 - 1, power=1)

window_size = config.feature_size[1]
clip_size = whole_clip.shape[1]

left_freq_map = np.zeros(shape=(512, clip_size))
right_freq_map = np.zeros(shape=(512, clip_size))

for (start, end) in windows(whole_clip, window_size=window_size, stride=window_size):
    frame_end = end
    whole = whole_clip[:,start:end]
    if (whole.shape[1] != window_size):
        frame_end = clip_size
        whole = whole_clip[:,clip_size-window_size:clip_size]

    whole_in = np.reshape(whole, (1, 1, config.feature_size[0], config.feature_size[1]))

    if config.use_gpu:
        whole_in = Variable(torch.from_numpy(whole_in)).cuda()
    else :
        whole_in = Variable(torch.from_numpy(whole_in))

    masks = net.module.predict(whole_in)    # 1 * 2 * 512 * 64

    masks = masks.data.cpu().numpy()

    # may optimize mask here
    audio_transfer.fix_mask(whole, masks[0,0,:,:], 60)
    audio_transfer.fix_mask(whole, masks[0,1,:,:], 60)

    left = masks[0,0,:,:] * whole
    right = masks[0,1,:,:] * whole
    
    print(start, frame_end)
    print(window_size)
    left_freq_map[:,start:frame_end] = left[:,window_size - (frame_end - start):window_size]
    right_freq_map[:,start:frame_end] = right[:,window_size - (frame_end - start):window_size]

    # apply transfer
    phase = audio_transfer.audio_to_phase(audio_clip, n_fft=config.feature_size[0] * 2 - 1, hop_length=512, win_length = config.feature_size[0] * 2 - 1)
    left_seq = audio_transfer.resynthesis(left_freq_map, phase, hop_length=512,win_length = config.feature_size[0] * 2 - 2)
    right_seq = audio_transfer.resynthesis(right_freq_map, phase, hop_length=512,win_length = config.feature_size[0] * 2 - 2)

    librosa.output.write_wav('left_'+file_name,left_seq, sr)
    librosa.output.write_wav('right_'+file_name,right_seq, sr)
