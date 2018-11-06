import librosa
import numpy as np

def audio_to_phase(audio, n_fft=2048):
    """
    Extract phase matrix of an audio sequence.
    :param audio: audio sequence
    :param n_fft: FFT(Fast Fourier Transform) window length
    :return: phase matrix
    """

    audio_stft = librosa.stft(audio, n_fft=n_fft)
    _, phase = librosa.magphase(audio_stft)
    return phase

def non_zero_min(mtx):
    min_elem = 1e10
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            elem = mtx[i][j].item()
            if elem != 0.0 and elem < min_elem:
                min_elem = elem
    return min_elem

def resynthesis(spec, phase, n_fft=2048):
    """
        Resynthesis the audio sequence from Mel Spectrogram with information given
          by the phase matrix.
        :param spec: np.ndarray, Mel Spectrogram
        :param phase: np.ndarray, phase matrix
        :param n_fft: parameter of fft, default 2048
        :return: audio sequence
    """

    nneg_spectrogram = np.maximum(spec, 0)
    complex_spectrogram = np.multiply(nneg_spectrogram, phase)
    audio_seq = librosa.istft(complex_spectrogram, win_length=n_fft)
    return audio_seq

def dumpf(spec, filename, dim):
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(spec.shape[0]):
            if dim == 2:
                for j in range(spec.shape[1]):
                    print(spec[i][j].item(), end=', ', file=f)
            elif dim == 1:
                print(spec[i].item(), end='', file=f)
            f.write('\n')

def test():
    b, sr = librosa.load('datas.wav', mono=False)
    voice = b[1, :]
    acc = b[0, :]
    librosa.output.write_wav('voice.wav', voice, sr)
    librosa.output.write_wav('acc.wav', acc, sr)

    full, sr = librosa.load('datas.wav', mono=True)
    phase = audio_to_phase(full)

    voice_spec, n_fft = librosa.core.spectrum._spectrogram(voice, n_fft=2048, power=1)
    acc_spec, n_fft = librosa.core.spectrum._spectrogram(acc, n_fft=2048, power=1)
    asq = resynthesis(voice_spec, phase, sr)
    acsq = resynthesis(acc_spec, phase, sr)
    librosa.output.write_wav('voice_asq.wav', asq, sr)
    librosa.output.write_wav('acc_asq.wav', acsq, sr)
