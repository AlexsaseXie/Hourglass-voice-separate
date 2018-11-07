'''
Part of the code modified from
https://github.com/andabi/music-source-separation
'''

import librosa
import numpy as np
import soundfile as sf
from . import andabi

class ModelConfig:
    SR = 8000
    L_FRAME = 1024
    L_HOP = 256



def estimate(mix_file, acc_file, voice_file):
    mix, sr = sf.read(mix_file)
    acc, voice = mix[:, 0], mix[:, 1]
    mix = librosa.to_mono(np.transpose(mix))
    pred_acc, _ = sf.read(acc_file)
    pred_voice, _ = sf.read(voice_file)

    nsdr, sdr, sdr_mix, sir, sar, length = bss_eval(mix, acc, voice, pred_acc, pred_voice)
    return nsdr, sir, sar, length

def estimate_batch(batch):
    # [accompaniment, voice]
    estimation = {
        'GNSDR': np.zeros(2, dtype=np.float64),
        'GSIR': np.zeros(2, dtype=np.float64),
        'GSAR': np.zeros(2, dtype=np.float64)
    }
    total_length = 0
    for mix_f, acc_f, voice_f in batch:
        nsdr, sir, sar, length = estimate(mix_f, acc_f, voice_f)
        estimation['GNSDR'] += nsdr * length
        estimation['GSIR'] += sir * length
        estimation['GSAR'] += sar * length
        total_length += length

    for k in estimation.keys():
        estimation[k] = estimation[k] / total_length

    return estimation


def get_wav(filename, sr=ModelConfig.SR):
    # src1_src2 = librosa.load(filename, sr=sr, mono=False)[0]
    mix, sr = sf.read(filename)
    # mixed = librosa.to_mono(src1_src2)
    src1, src2 = mix[:, 0], mix[:, 1]
    return mix, src1, src2

def to_wav_file(mag, phase, len_hop=ModelConfig.L_HOP):
    stft_maxrix = get_stft_matrix(mag, phase)
    return np.array(librosa.istft(stft_maxrix, hop_length=len_hop))

def to_spec(wav, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP):
    return librosa.stft(wav, n_fft=len_frame, hop_length=len_hop)

def get_stft_matrix(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)

def write_wav(data, path, sr=ModelConfig.SR, format='wav', subtype='PCM_16'):
    sf.write(path, data, sr, format=format, subtype=subtype)

def bss_eval(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    length = pred_src1_wav.shape[0]
    src1_wav = src1_wav[:length]
    src2_wav = src2_wav[:length]
    mixed_wav = mixed_wav[:length]
    sdr, sir, sar, _ = andabi.bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), compute_permutation=True)
    sdr_mixed, _, _, _ = andabi.bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), compute_permutation=True)
    # sdr, sir, sar, _ = bss_eval_sources(src2_wav,pred_src2_wav, False)
    # sdr_mixed, _, _, _ = bss_eval_sources(src2_wav,mixed_wav, False)
    nsdr = sdr - sdr_mixed
    return nsdr, sdr, sdr_mixed, sir, sar, length

def bss_eval_sdr(src1_wav, pred_src1_wav):
        len_cropped = pred_src1_wav.shape[0]
        src1_wav = src1_wav[:len_cropped]

        sdr, _, _, _ = andabi.bss_eval_sources(src1_wav,
                                            pred_src1_wav, compute_permutation=True)
        return sdr