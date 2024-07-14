import os
import numpy as np
from scipy.signal import hamming
import scipy.fftpack as fft
import librosa

def pncc(y, sr, n_fft=512, n_mels=20, n_pncc=13, fmin=0, fmax=None, win_length=0.032, hop_length=0.016, preemph=0.97):
    # 预加重
    y = np.append(y[0], y[1:] - preemph * y[:-1])
    # 短时傅里叶变换 (STFT)
    D = librosa.stft(y, n_fft=n_fft, hop_length=int(hop_length * sr), win_length=int(win_length * sr),
                     window=hamming(int(win_length * sr)))
    # 功率谱
    S = np.abs(D) ** 2
    # Mel 滤波器组
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    # Mel 频谱
    mel_S = np.dot(mel_basis, S)
    # 功率归一化
    mel_S = mel_S / np.max(mel_S, axis=0, keepdims=True)
    # 长时域平均
    smoothed_mel_S = np.apply_along_axis(lambda m: np.convolve(m, np.ones(4) / 4, mode='same'), axis=1, arr=mel_S)
    # 非线性变换
    mel_S = smoothed_mel_S ** (1. / 15)
    # 动态范围调整
    mel_S = np.log10(np.maximum(mel_S, np.finfo(float).eps))
    # 离散余弦变换 (DCT)
    pnccs = fft.dct(mel_S, axis=0, type=2, norm='ortho')[:n_pncc]

    return pnccs

# 提取 UBM特征
fea_train_path = "fea_pncc/TRAIN"
os.makedirs(fea_train_path, exist_ok=True)

file_lines = np.loadtxt('ubm_wav.scp', dtype='str', delimiter=" ")
files = file_lines[:, 0]
spk_ids = file_lines[:, 1]
utt_ids = file_lines[:, 2]

for file, spk, utt in zip(files, spk_ids, utt_ids):
    # 读取音频文件
    y, fs = librosa.load(file, sr=None, mono=True)

    # 进行PNCC特征的提取
    pnccs = pncc(y=y, sr=fs)

    file_fea = os.path.join(fea_train_path, spk + "_" + utt + ".npy")
    np.save(file=file_fea, arr=pnccs)
    print("save_file ", file_fea)

# 提取 test数据特征
fea_test_path = "fea_pncc/TEST"
os.makedirs(fea_test_path, exist_ok=True)

file_lines = np.loadtxt('test.scp', dtype='str', delimiter=" ")
files = file_lines[:, 0]
spk_ids = file_lines[:, 1]
utt_ids = file_lines[:, 2]

for file, spk, utt in zip(files, spk_ids, utt_ids):
    # 读取音频文件
    y, fs = librosa.load(file, sr=None, mono=True)

    # 进行PNCC特征的提取
    pnccs = pncc(y=y, sr=fs)

    file_fea = os.path.join(fea_test_path, spk + "_" + utt + ".npy")
    np.save(file=file_fea, arr=pnccs)
    print("save_file ", file_fea)

print('Done')