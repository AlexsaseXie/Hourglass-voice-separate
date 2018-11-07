'''
Part of the code modified from
https://github.com/andabi/music-source-separation
'''

import librosa
import numpy as np
import soundfile as sf
from . import andabi
from src.utils import audio_transfer

class ModelConfig:
    SR = 8000
    L_FRAME = 1024
    L_HOP = 256



def estimate(mix, acc, voice, pred_acc, pred_voice):
    # mix, sr = sf.read(mix_file)
    # acc, voice = mix[:, 0], mix[:, 1]
    # mix = librosa.to_mono(np.transpose(mix))
    # pred_acc, _ = sf.read(acc_file)
    # pred_voice, _ = sf.read(voice_file)

    nsdr, sdr, sdr_mix, sir, sar, length = bss_eval(mix, acc, voice, pred_acc, pred_voice)
    return nsdr, sir, sar, length


def estimate_batch(whole_batch, left_batch, right_batch, mask_batch, phase_batch, phase_acc_batch, phase_voice_batch, batch_size, fixing_mask):
    # matrix size: batch * 1(2) * 512 * 64
    # [accompaniment, voice]
    estimation = {
        'GNSDR': np.zeros(2, dtype=np.float64),
        'GSIR': np.zeros(2, dtype=np.float64),
        'GSAR': np.zeros(2, dtype=np.float64)
    }
    total_length = 0
    for i in range(batch_size):
        whole_spec = whole_batch[i, 0, :, :]
        acc_spec = left_batch[i, 0, :, :]
        voice_spec = right_batch[i, 0, :, :]
        mask_acc = mask_batch[i, 0, :, :]
        mask_voice = mask_batch[i, 1, :, :]
        phase = phase_batch[i, 0, :, :]
        phase_acc = phase_acc_batch[i, 0, :, :]
        phase_voice = phase_voice_batch[i, 0, :, :]

        if fixing_mask:
            mask_voice = audio_transfer.fix_mask(whole_spec, mask_voice, 100)

        whole_seq = audio_transfer.resynthesis(whole_spec, phase, 512, 1022)
        acc_seq = audio_transfer.resynthesis(acc_spec, phase_acc, 512, 1022)
        voice_seq = audio_transfer.resynthesis(voice_spec, phase_voice, 512, 1022)
        pred_acc_seq = audio_transfer.resynthesis(mask_acc * whole_spec, phase, 512, 1022)
        pred_voice_seq = audio_transfer.resynthesis(mask_voice * whole_spec, phase, 512, 1022)

        nsdr, sir, sar, length = bss_eval(whole_seq, acc_seq, voice_seq, pred_acc_seq, pred_voice_seq)
        estimation['GNSDR'] += nsdr * length
        estimation['GSIR'] += sir * length
        estimation['GSAR'] += sar * length
        total_length += length

    return estimation, total_length

    # for mix_f, acc_f, voice_f in batch:
    #     nsdr, sir, sar, length = estimate(mix_f, acc_f, voice_f)
    #     estimation['GNSDR'] += nsdr * length
    #     estimation['GSIR'] += sir * length
    #     estimation['GSAR'] += sar * length
    #     total_length += length
    #
    # for k in estimation.keys():
    #     estimation[k] = estimation[k] / total_length
    #
    # return estimation


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
    return nsdr, sir, sar, length

def bss_eval_sdr(src1_wav, pred_src1_wav):
        len_cropped = pred_src1_wav.shape[0]
        src1_wav = src1_wav[:len_cropped]

        sdr, _, _, _ = andabi.bss_eval_sources(src1_wav,
                                            pred_src1_wav, compute_permutation=True)
        return sdr