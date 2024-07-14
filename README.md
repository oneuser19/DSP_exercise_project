# DSP_exercise_project
一个数字信号处理练习项目。
基于GMM-UBM模型的声纹识别系统。分别使用FBANK、MFCC和PNCC三种特征提取方法。
1、下载 TIMIT 数据集放在根目录
2、运行 train_spk_model.py 生成 ubm_wav.scp，test.scp
3、运行  val_enrollment_scp.py  生成 enrollment.scp，var.scp
4、运行 feature_ex_Fbank.py 生成 fbank 特征，运行 feature_extract_MFCC.py 生成 mfcc 特征，运行 feature_extract_pncc.py 生成 pncc 特征
5、运行 train_UBM.py，用chooes()函数选择参数 fbank，mfcc，pncc，训练对应特征的UBM
6、运行 train_spk_model.py，用chooes()函数选择参数 fbank，mfcc，pncc，生成对应特征的说话人GMM
7、运行 eval_score.py，用chooes()函数选择参数 fbank，mfcc，pncc，进行测试打分计算EER

A digital signal processing exercise project.
Voiceprint recognition system based on GMM-UBM model. Three feature extraction methods, FBANK, MFCC and PNCC, were used, respectively.
1. Download the TIMIT dataset and place it in the root directory
2. Run train_spk_model.py to generate ubm_wav.scp, test.scp
3. Run val_enrollment_scp.py to generate enrollment.scp,var.scp
4. Run feature_ex_Fbank.py to generate fbank features, run feature_extract_MFCC.py to generate mfcc features, and run feature_extract_pncc.py to generate pncc features
5. Run the train_UBM.py, use the chooes() function to select the parameters fbank, mfcc, pncc, and train the UBM of the corresponding features
6. Run train_spk_model.py, use the chooes() function to select the parameters fbank, mfcc, pncc, and generate the speaker GMM of the corresponding features
7. Run the eval_score.py, use the chooes() function to select the parameters fbank, mfcc, and pncc to score the test and calculate the EER
