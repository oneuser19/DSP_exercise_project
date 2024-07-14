import  os
# 生成 TRAIN.scp
root_path= "./TIMIT/TRAIN"
with open("ubm_wav.scp",'wt') as f:
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file_path in filenames:
            if file_path.endswith(".WAV"):
                full_name= os.path.join(dirpath,file_path)
                speak_id = os.path.split(dirpath)[-1]
                utt_id = file_path.split(".")[0]
                f.write("%s %s %s\n"%(full_name,speak_id,utt_id))
                print("%s %s %s"%(full_name,speak_id,utt_id))
print('Get ubm_wav.scp')
# 生成 TEST.scp
root_path= "./TIMIT/TEST"
with open("test.scp",'wt') as f:
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file_path in filenames:
            if file_path.endswith(".WAV"):
                full_name= os.path.join(dirpath,file_path)
                speak_id = os.path.split(dirpath)[-1]
                utt_id = file_path.split(".")[0]
                f.write("%s %s %s\n"%(full_name,speak_id,utt_id))
                print("%s %s %s"%(full_name,speak_id,utt_id))
print('Get test.scp')