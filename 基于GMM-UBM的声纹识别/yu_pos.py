import os
import librosa
import soundfile as sf
# 定义函数执行信号处理操作
def process_wav_file(file_path):
    # 加载音频文件
    y, fs = librosa.load(file_path, sr=16000)
    # 修剪音频文件
    yt, index = librosa.effects.trim(y, top_db=30)
    # 分割和重组音频文件
    intervals = librosa.effects.split(yt, top_db=20)
    y_remix = librosa.effects.remix(yt, intervals)
    # 覆盖原文件
    sf.write(file_path, y_remix, fs)
    print(f"Processed and saved: {file_path}")
# 遍历指定路径下所有 .WAV 文件
def process_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".wav"):  # 检查文件是否以 .wav 结尾
                file_path = os.path.join(root, file)
                process_wav_file(file_path)
# 定义两个路径
root_paths = ["./TIMIT/TRAIN", "./TIMIT/TEST"]
# 遍历并处理每个路径下的文件
for root_path in root_paths:
    process_files_in_directory(root_path)

print('Down')
