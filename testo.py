2
# coding: utf-8

# In[43]:

# import csv
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import time

# for root, dirs, files in os.walk(r"."):
#     print(1)
# for file in files:
#     if file[-3:] != 'wav':
#         continue
#     print(file)
#     y, sr = librosa.load(os.path.join(root, file))
#     # Let's make and display a mel-scaled power (energy-squared) spectrogram
#     S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

#     # Convert to log scale (dB). We'll use the peak power as reference.
#     log_S = librosa.power_to_db(S)

#     # Make a new figure
#     fig = plt.figure(figsize=(12,4))
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)

#     # Display the spectrogram on a mel scale
#     # sample rate and hop length parameters are used to render the time axis
#     librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

#     # Make the figure layout compact

#     #plt.show()
#     plt.savefig(os.path.splitext(file)[0]+'.png')
#     plt.close()

#print count


# In[44]:

def runtest():
    name = 'abjones_1_01'
    filepath = name + '.wav'
    y, sr = librosa.load(filepath, sr=None, mono=True)
    print(sr)
    print(y.shape)

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)

    print(type(S))

    log_S = librosa.power_to_db(S)

    print(log_S.shape)

    fig = plt.figure(figsize=(12,4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.savefig(name + '.png')

    librosa.output.write_wav(name + '_new.wav', y, sr)

    plt.close()

if __name__ == '__main__':
    runtest()
