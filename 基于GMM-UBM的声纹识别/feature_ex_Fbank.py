import librosa
import os
import numpy as np

# 提取UBM特征
fea_train_path = "fea_fbank/TRAIN"
os.makedirs(fea_train_path, exist_ok=True)

file_lines = np.loadtxt('ubm_wav.scp', dtype='str', delimiter=" ")
files = file_lines[:, 0]
spk_ids = file_lines[:, 1]
utt_ids = file_lines[:, 2]

for file, spk, utt in zip(files, spk_ids, utt_ids):
    # 读取音频文件
    y, fs = librosa.load(file, sr=None, mono=True)
    # 应用预加重滤波器
    y = librosa.effects.preemphasis(y)

    # 参数设置
    win_length = 320
    hop_length = 160
    n_fft = 512
    n_mels = 20
    # 提取FBank特征
    fbank = librosa.feature.melspectrogram(y=y,
                                           sr=fs,
                                           n_fft=n_fft,
                                           win_length=win_length,
                                           hop_length=hop_length,
                                           n_mels=n_mels)
    # 转换为dB
    fbank_db = librosa.power_to_db(fbank, ref=np.max)

    # 增加动态特征
    fbank_delta = librosa.feature.delta(fbank_db)
    fbank_delta2 = librosa.feature.delta(fbank_db, order=2)
    # 拼接生成最终的FBank特征
    fea_fbank = np.concatenate([fbank_db, fbank_delta, fbank_delta2], axis=0)

    # 保存特征
    file_fea = os.path.join(fea_train_path, spk + "_" + utt + ".npy")
    np.save(file=file_fea, arr=fea_fbank)
    print("save_file ", file_fea)

# 提取测试数据特征
fea_test_path = "fea_fbank/TEST"
os.makedirs(fea_test_path, exist_ok=True)

file_lines = np.loadtxt('test.scp', dtype='str', delimiter=" ")
files = file_lines[:, 0]
spk_ids = file_lines[:, 1]
utt_ids = file_lines[:, 2]

for file, spk, utt in zip(files, spk_ids, utt_ids):
    # 读取音频文件
    y, fs = librosa.load(file, sr=None, mono=True)
    # 应用预加重滤波器
    y = librosa.effects.preemphasis(y)
    # 参数设置
    win_length = 320
    hop_length = 160
    n_fft = 512
    n_mels = 20
    # 提取FBank特征
    fbank = librosa.feature.melspectrogram(y=y,
                                           sr=fs,
                                           n_fft=n_fft,
                                           win_length=win_length,
                                           hop_length=hop_length,
                                           n_mels=n_mels)

    # 转换为dB
    fbank_db = librosa.power_to_db(fbank, ref=np.max)
    # 增加动态特征
    fbank_delta = librosa.feature.delta(fbank_db)
    fbank_delta2 = librosa.feature.delta(fbank_db, order=2)
    # 拼接生成最终的FBank特征
    fea_fbank = np.concatenate([fbank_db, fbank_delta, fbank_delta2], axis=0)

    # 保存特征
    file_fea = os.path.join(fea_test_path, spk + "_" + utt + ".npy")
    np.save(file=file_fea, arr=fea_fbank)
    print("save_file ", file_fea)

print('Done')