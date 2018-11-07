import numpy as np
import os
import librosa
import numpy as np
from matplotlib import pyplot as plt
from ..utils import audio_transfer

class Generator:
    """
    file_path: wavs file path
    feature_size: feature_size
    """
    def __init__(self, file_path='data/train/', feature_size=[512, 64]):
        self.files = []
        for file in os.listdir(file_path):
            if (file != '.placeholder'):
                tmp_path = os.path.join(file_path, file)
                self.files.append(tmp_path)

        self.feature_size = feature_size

        self.whole_mel = np.zeros(shape=(1, feature_size[0], feature_size[1]))
        self.left_mel = np.zeros(shape=(1, feature_size[0], feature_size[1]))
        self.right_mel = np.zeros(shape=(1, feature_size[0], feature_size[1]))

        # load files
        for fp in self.files:
            mono_y, _ = librosa.load(fp, sr=None, mono=True)
            left_right_y, _ = librosa.load(fp, sr=None, mono=False)

            using_nfft = self.feature_size[0] * 2 - 1
            new_whole_mel, _ = librosa.core.spectrum._spectrogram(mono_y, n_fft=using_nfft, power=1)
            new_left_mel, _ = librosa.core.spectrum._spectrogram(left_right_y[0], n_fft=using_nfft, power=1)
            new_right_mel, _ = librosa.core.spectrum._spectrogram(left_right_y[1], n_fft=using_nfft, power=1)

            self.insert_mel_list(new_whole_mel, mode = 1)
            self.insert_mel_list(new_left_mel, mode = 2)
            self.insert_mel_list(new_right_mel, mode = 3)

        self.whole_mel = self.whole_mel[1:,:,:]
        self.left_mel = self.left_mel[1:,:,:]
        self.right_mel = self.right_mel[1:,:,:]
        pass 

    def windows(self, data, window_size, stride):
        """
        data : H * W, window on W 
        """
        start = 0
        while start + window_size < data.shape[1]:
            yield start, start + window_size
            start += stride

    def insert_mel_list(self, data, mode):
        """
        data : H * W, window on W
        mode : distinguish to insert into 3 categories
        """
        window_size = self.feature_size[1]
        for (start, end) in self.windows(data, window_size, stride= window_size // 2):
            if (end - start == window_size):
                tmp = data[:, start:end]
                if (mode == 1):
                    self.whole_mel = np.append(self.whole_mel, [tmp], axis = 0)
                elif (mode == 2):
                    self.left_mel = np.append(self.left_mel, [tmp], axis = 0)
                elif mode == 3:
                    self.right_mel = np.append(self.right_mel, [tmp], axis = 0)

    def get_file_data(self, batch_size: int, if_randomize=True):
        """
        batch_size: 
        """
        # The last label corresponds to the stop symbol and the first one to
        # start symbol.


        #open & save the files

        clip_count = self.whole_mel.shape[0]

        while True:
            # Random things to select random indices
            IDS = np.arange(clip_count)
            if if_randomize:
                np.random.shuffle(IDS)
            for rand_id in range(0, clip_count - batch_size,
                                 batch_size):
                file_ids = IDS[rand_id:rand_id + batch_size]

                wholes = self.whole_mel[file_ids, :, :].astype(dtype=np.float32)
                lefts = self.left_mel[file_ids, :, :].astype(dtype=np.float32)
                rights = self.right_mel[file_ids, :, :].astype(dtype=np.float32)

                yield [wholes, lefts, rights]


class TestGenerator:
    """
    file_path: wavs file path
    feature_size: feature_size
    """
    def __init__(self, file_path='data/train/', feature_size=[512, 64]):
        self.files = []
        for file in os.listdir(file_path):
            if (file != '.placeholder'):
                tmp_path = os.path.join(file_path, file)
                self.files.append(tmp_path)

        self.feature_size = feature_size

        self.whole_mel = np.zeros(shape=(1, feature_size[0], feature_size[1]))
        self.left_mel = np.zeros(shape=(1, feature_size[0], feature_size[1]))
        self.right_mel = np.zeros(shape=(1, feature_size[0], feature_size[1]))
        self.phase = np.zeros(shape=(1, feature_size[0], feature_size[1]))
        self.phase_acc = np.zeros(shape=(1, feature_size[0], feature_size[1]))
        self.phase_voice = np.zeros(shape=(1, feature_size[0], feature_size[1]))

        # load files
        for fp in self.files:
            mono_y, _ = librosa.load(fp, sr=None, mono=True)
            left_right_y, _ = librosa.load(fp, sr=None, mono=False)

            using_nfft = self.feature_size[0] * 2 - 1
            new_whole_mel, _ = librosa.core.spectrum._spectrogram(mono_y, n_fft=using_nfft, power=1)
            new_phase = audio_transfer.audio_to_phase(mono_y, using_nfft, 512, using_nfft)
            new_phase_acc = audio_transfer.audio_to_phase(left_right_y[0], using_nfft, 512, using_nfft)
            new_phase_voice = audio_transfer.audio_to_phase(left_right_y[1], using_nfft, 512, using_nfft)
            new_left_mel, _ = librosa.core.spectrum._spectrogram(left_right_y[0], n_fft=using_nfft, power=1)
            new_right_mel, _ = librosa.core.spectrum._spectrogram(left_right_y[1], n_fft=using_nfft, power=1)

            self.insert_mel_list(new_whole_mel, mode = 1)
            self.insert_mel_list(new_left_mel, mode = 2)
            self.insert_mel_list(new_right_mel, mode = 3)
            self.insert_mel_list(new_phase, mode=4)
            self.insert_mel_list(new_phase_acc, mode=5)
            self.insert_mel_list(new_phase_voice, mode=6)

        self.whole_mel = self.whole_mel[1:,:,:]
        self.left_mel = self.left_mel[1:,:,:]
        self.right_mel = self.right_mel[1:,:,:]
        self.phase = self.phase[1:, :, :]
        self.phase_acc = self.phase_acc[1:, :, :]
        self.phase_voice = self.phase_voice[1:, :, :]
        pass 

    def windows(self, data, window_size, stride):
        """
        data : H * W, window on W 
        """
        start = 0
        while start + window_size < data.shape[1]:
            yield start, start + window_size
            start += stride

    def insert_mel_list(self, data, mode):
        """
        data : H * W, window on W
        mode : distinguish to insert into 3 categories
        """
        window_size = self.feature_size[1]
        for (start, end) in self.windows(data, window_size, stride= window_size // 2):
            if (end - start == window_size):
                tmp = data[:, start:end]
                if (mode == 1):
                    self.whole_mel = np.append(self.whole_mel, [tmp], axis = 0)
                elif (mode == 2):
                    self.left_mel = np.append(self.left_mel, [tmp], axis = 0)
                elif mode == 3:
                    self.right_mel = np.append(self.right_mel, [tmp], axis = 0)
                elif mode == 4:
                    self.phase = np.append(self.phase, [tmp], axis = 0)
                elif mode == 5:
                    self.phase_acc = np.append(self.phase_acc, [tmp], axis=0)
                elif mode == 6:
                    self.phase_voice = np.append(self.phase_voice, [tmp], axis=0)

    def get_file_data(self, batch_size: int, if_randomize=True):
        """
        batch_size: 
        """
        # The last label corresponds to the stop symbol and the first one to
        # start symbol.


        #open & save the files

        clip_count = self.whole_mel.shape[0]

        while True:
            # Random things to select random indices
            IDS = np.arange(clip_count)
            if if_randomize:
                np.random.shuffle(IDS)
            for rand_id in range(0, clip_count - batch_size,
                                 batch_size):
                file_ids = IDS[rand_id:rand_id + batch_size]

                wholes = self.whole_mel[file_ids, :, :].astype(dtype=np.float32)
                lefts = self.left_mel[file_ids, :, :].astype(dtype=np.float32)
                rights = self.right_mel[file_ids, :, :].astype(dtype=np.float32)
                phases = self.phase[file_ids, :, :].astype(dtype=np.complex64)
                phases_acc = self.phase_acc[file_ids, :, :].astype(dtype=np.complex64)
                phases_voice = self.phase_voice[file_ids, :, :].astype(dtype=np.complex64)

                yield [wholes, lefts, rights, phases, phases_acc, phases_voice]

        

        