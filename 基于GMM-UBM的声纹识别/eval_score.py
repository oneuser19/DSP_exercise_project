import numpy as np
import os
import joblib
import sklearn.metrics
import pandas as pd

def getscore(ubm, gmm, data):
    score_ubm = ubm.score(data)
    score_gmm = gmm.score(data)
    return score_gmm - score_ubm

def compute_eer(label, pred, pos_label=1):
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, pos_label=pos_label)
    fnr = 1 - tpr

    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    eer = (eer_1 + eer_2) / 2
    return eer, eer_threshold

def choose(a):
    global path_fea, path_model
    if a == 'fbank':
        path_fea = 'fea_fbank/TEST'
        path_model = 'models_fbank'
    if a == 'mfcc':
        path_fea = 'fea_mfcc/TEST'
        path_model = 'models_mfcc'
    if a == 'pncc':
        path_fea = 'fea_pncc/TEST'
        path_model = 'models_pncc'


if __name__ == "__main__":
    choose('mfcc')
    # 加载UBM
    ubm = joblib.load(os.path.join(path_model, 'ubm.model'))

    # 加载验证数据
    file_lines = np.loadtxt("var.scp", dtype='str', delimiter=" ")
    spks_true = file_lines[:, 1]
    utts = file_lines[:, 2]
    spks_var = file_lines[:, 3]
    labs = file_lines[:, 4]
    labs = [int(lab) for lab in labs]
    scores = []

    results = []

    for spk_true, utt, spk_var, lab in zip(spks_true, utts, spks_var, labs):
        file_fea = os.path.join(path_fea, spk_true + '_' + utt + '.npy')
        data = np.load(file_fea).T

        gmm = joblib.load(os.path.join(path_model, spk_var + '.model'))
        score = getscore(ubm, gmm, data)
        scores.append(score)
        print(spk_true, ' ', spk_var, ' ', "%.3f" % (score))

        results.append([spk_true, spk_var, score])

    eer, thred = compute_eer(labs, scores, pos_label=1)
    print("等错误率：")
    print(eer)
    print("对应阈值")
    print(thred)

    # 将结果保存到 Excel 文件
    if path_fea == 'fea_fbank/TEST':
        df = pd.DataFrame(results, columns=['spk_true', 'spk_var', 'score'])
        df.to_excel('results_fbank.xlsx', index=False)
        print("结果已保存到 results_fbank.xlsx 文件")
    if path_fea == 'fea_mfcc/TEST':
        df = pd.DataFrame(results, columns=['spk_true', 'spk_var', 'score'])
        df.to_excel('results_mfcc.xlsx', index=False)
        print("结果已保存到 results_mfcc.xlsx 文件")
    if path_fea == 'fea_pncc/TEST':
        df = pd.DataFrame(results, columns=['spk_true', 'spk_var', 'score'])
        df.to_excel('results_pncc.xlsx', index=False)
        print("结果已保存到 results_pncc.xlsx 文件")
