import numpy as np
import random

# 读取 TEST.scp
test_lines= np.loadtxt("test.scp",dtype='str',delimiter=" ")
files = test_lines[:,0]
spk_ids = test_lines[:,1]
utt_ids = test_lines[:,2]

unique_spks = np.unique(spk_ids)
N_spk = len(unique_spks)
f_enrollment = open("enrollment.scp",'wt')
f_var  = open("var.scp",'wt')
rand_sel = [i for i in range(N_spk)]

for spk in unique_spks:
    index = np.where(spk_ids==spk)[0].tolist()
    N_utt = len(index)
    N_enroll = int(N_utt*0.7)
    N_test = N_utt-N_enroll
    # 随机选取7条作为注册语音，3条作为测试语音
    random.shuffle(index)

    for i, shuffle_index in enumerate(index) :
        full_name = files[shuffle_index]
        utt_id = utt_ids[shuffle_index]
        # 写 注册scp文件
        if i < N_enroll:
            f_enrollment.write("%s %s %s\n"%(full_name,spk,utt_id))
        # 写 测试 scp文件
        else:
            # 写入一条正确的测试语音
            test_id = spk
            f_var.write("%s %s %s %s 1\n"%(full_name,spk,utt_id,test_id))
            # 再随机选取10个其他的说话人
            random.shuffle(rand_sel)
            for i_sel in range(10):
                test_id = unique_spks[rand_sel[i_sel]]
                if test_id == spk:
                    f_var.write("%s %s %s %s 1\n"%(full_name,spk,utt_id,test_id))
                else:
                    f_var.write("%s %s %s %s 0\n"%(full_name,spk,utt_id,test_id))

f_var.close()
print('Get var.scp')
f_enrollment.close()
print('Get enrollment.scp')

