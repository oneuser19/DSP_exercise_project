import librosa
import os
import numpy as np

# 提取 UBM特征

fea_train_path = "fea_mfcc/TRAIN"
os.makedirs(fea_train_path,exist_ok=True)

file_lines = np.loadtxt('ubm_wav.scp',dtype='str',delimiter=" ")
files= file_lines[:,0]
spk_ids = file_lines[:,1]
utt_ids = file_lines[:,2]

for file,spk,utt in zip(files,spk_ids,utt_ids):
    # 读取音频文件
    y,fs = librosa.load(file,sr=None, mono=True)
    # 应用预加重滤波器
    y = librosa.effects.preemphasis(y)
    # 进行MFCC特征的提取
    raw_mfcc = librosa.feature.mfcc(y=y,
                                   sr=fs,
                                   n_mfcc=19,
                                   n_fft=512,
                                   hop_length=160,
                                   win_length=320,
                                   n_mels=20
                                   )
    # 增加动态特征
    raw_mfcc = librosa.util.normalize(raw_mfcc)
    mfcc_delta = librosa.feature.delta(raw_mfcc)
    mfcc_delta2 = librosa.feature.delta(raw_mfcc, order=2)
    # 拼接生成最终的MFCC特征
    fea_mfcc = np.concatenate([raw_mfcc,mfcc_delta,mfcc_delta2],axis=0)

    file_fea = os.path.join(fea_train_path,spk+"_"+utt+".npy")
    np.save(file=file_fea,arr=fea_mfcc)
    print("save_file ",file_fea)


# 提取 test数据特征

fea_train_path = "fea_mfcc/TEST"
os.makedirs(fea_train_path,exist_ok=True)

file_lines = np.loadtxt('test.scp',dtype='str',delimiter=" ")
files= file_lines[:,0]
spk_ids = file_lines[:,1]
utt_ids = file_lines[:,2]

for file,spk,utt in zip(files,spk_ids,utt_ids):
    # 读取音频文件
    y,fs = librosa.load(file,sr=None, mono=True)
    # 应用预加重滤波器
    y = librosa.effects.preemphasis(y)
    # 进行MFCC特征的提取
    raw_mfcc = librosa.feature.mfcc(y=y,
                                   sr=fs,
                                   n_mfcc=19,
                                   n_fft=512,
                                   hop_length=160,
                                   win_length=320,
                                   n_mels=20
                                   )
    # 增加动态特征
    raw_mfcc = librosa.util.normalize(raw_mfcc)
    mfcc_delta = librosa.feature.delta(raw_mfcc)
    mfcc_delta2 = librosa.feature.delta(raw_mfcc, order=2)
    # 拼接生成最终的MFCC特征
    fea_mfcc = np.concatenate([raw_mfcc,mfcc_delta,mfcc_delta2],axis=0)

    file_fea = os.path.join(fea_train_path,spk+"_"+utt+".npy")
    np.save(file=file_fea,arr=fea_mfcc)
    print("save_file ",file_fea)

print('Done')