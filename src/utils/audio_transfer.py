import librosa
import numpy as np
import sys
from queue import Queue

def audio_to_phase(audio, n_fft, hop_length, win_length):
    """
        Extract phase matrix of an audio sequence.
        :param audio: audio sequence
        :param n_fft: FFT(Fast Fourier Transform) window length
        :return: phase matrix
    """

    audio_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    _, phase = librosa.magphase(audio_stft, power=1)
    return phase

def resynthesis(spec, phase, hop_length, win_length):
    """
        Resynthesis the audio sequence from Mel Spectrogram with information given
          by the phase matrix.
        :param spec: np.ndarray, Mel Spectrogram
        :param phase: np.ndarray, phase matrix, each element is represented as
          complex form (modulus equals 1), not angle
        :param n_fft: parameter of fft, default 2048
        :return: audio sequence
    """

    nneg_spectrogram = np.maximum(spec, 0)
    complex_spectrogram = np.multiply(nneg_spectrogram, phase)
    audio_seq = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=win_length)
    return audio_seq

def fix_mask(spec, mask, R):
    """
        Get modified mask using NMF and clustering
        :param spec: np.ndarray(C, M), STFT spectrogram of input audio
        :param mask: np.ndarray(C, M), original mask matrix
        :param R: int, feature number, 160 in paper
        :return: modified mask, actually M0 || M1
    """
    C, M = spec.shape
    W, H = librosa.decompose.decompose(spec, R)
    print(W.shape)
    print(H.shape)
    # W: C * R
    # H: R * M
    layers = np.empty((C, M, R), dtype=np.float32)
    for r in range(R):
        np.multiply(W[:, r].reshape(-1, 1), H[r, :], out=layers[:, :, r])

    prominent_component = np.empty((C, M), dtype=np.int64)
    prominent_mask = np.empty((C, M), dtype=np.float32)
    for c in range(C):
        for m in range(M):
            prominent_mask[c][m] = np.max(layers[c, m, :])
            prominent_component[c][m] = np.argmax(layers[c, m, :])

    clustering(prominent_mask, prominent_component)
    return np.maximum(mask, prominent_mask)


def clustering(mask, component):
    """
        Modify mask matrix, let it clustered by information in component
        :param mask: np.ndarray(C, M), original mask matrix
        :param component: np.ndarray(C, M)[dytpe=np.int64], component matrix
        :return: None
    """
    C, M = mask.shape
    visit = np.zeros((C, M), dtype=bool)
    cluster = set()
    for i in range(C):
        for j in range(M):
            if visit[i][j].item() is False:
                cluster.clear()
                q = Queue()
                q.put((i, j), False)
                visit[i][j] = True
                allsum = seek(i, j,
                              C=C,
                              M=M,
                              visit=visit,
                              label=component[i][j].item(),
                              mask=mask,
                              comp=component,
                              q=q,
                              cluster=cluster)

                avg = allsum / len(cluster)

                for ci, cj in cluster:
                    mask[ci][cj] = avg


def valid(i, j, **kwargs):
    if i < 0 or i >= kwargs['C'] or j < 0 or j >= kwargs['M']:
        return False
    if kwargs['visit'][i][j].item() is True:
        return False
    if kwargs['comp'][i][j].item() != kwargs['label']:
        return False
    return True

def seek(i, j, **kwargs):
    cursum = 0
    vec = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while not kwargs['q'].empty():
        i, j = kwargs['q'].get(False)
        cursum += kwargs['mask'][i][j].item()
        kwargs['cluster'].add((i, j))
        for v in vec:
            nexti, nextj = i + v[0], j + v[1]
            if valid(nexti, nextj, **kwargs):
                kwargs['visit'][nexti][nextj] = True
                kwargs['q'].put((nexti, nextj))

    return cursum


def test():
    using_nfft = 1023
    full, sr = librosa.load('datas.wav', mono=True)
    full_phase = audio_to_phase(full, using_nfft)
    voice = librosa.load('datas.wav', mono=False)[0][1, :]
    voice_phase = audio_to_phase(voice, using_nfft)
    full_spec, n_fft = librosa.core.spectrum._spectrogram(full, n_fft=using_nfft, hop_length=using_nfft // 4, power=1)
    voice_spec, n_fft = librosa.core.spectrum._spectrogram(voice, n_fft=using_nfft, hop_length=using_nfft // 4, power=1)

    voice_origin_mask = np.divide(voice_spec, full_spec)
    voice_fix_mask = fix_mask(full_spec, voice_origin_mask, 160)
    output_spec = np.multiply(full_spec, voice_fix_mask)
    voice_seq = resynthesis(output_spec, voice_phase, using_nfft)
    librosa.output.write_wav('voice_fixmask_seq.wav', voice_seq, sr)