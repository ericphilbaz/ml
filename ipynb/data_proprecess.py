#-*- coding:utf-8 -*-
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import  pandas as pd
import pywt
def prepro(d_path,d_path1,length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28):
    print('train_path:', d_path)
    # print('test_path:', d_path1)
    filenames_train = os.listdir(d_path)
    filenames_test = os.listdir(d_path)
    def capture(path_train,filenames):
        """读取csv文件，返回数据
        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(path_train, i)
            # --------------------------读取csv--------------------------
            file = np.loadtxt(file_path, skiprows=1)
            files_2 = pywt.WaveletPacket(data=file, wavelet='db3',mode='symmetric', maxlevel=3)
            file_a = files_2['a'].data
            file_d = files_2['d'].data
            file_all = pd.DataFrame()
            file_all['a']=file_a
            file_all['d']=file_d
            files.update(dict(zip([i], [file_all])))
        return files

    def slice_enc(data, d_path_add, slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.
        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        # keys = data.keys()
        keys = os.listdir(d_path_add)
        # print(keys)
        Train_Samples = {}
        Test_Samples = {}
        # print('*'*50)
        # print(data)
        # print('*'*50)
        # 从end_index前抓取训练集数据
        for i in keys:
            slice_data = data[i]
            # print(data[i])
            all_lenght = len(slice_data)
            # print(all_lenght)
            end_index = int(all_lenght * (1 - slice_rate))
            # print(end_index)
            samp_train = int(number * (1 - slice_rate))  # 700
            # print(samp_train)
            Train_sample = []
            Test_Sample = []
            if enc:
                enc_time = length // enc_step
                samp_step = 0  # 用来计数Train采样次数
                #                 print("."*20,'数据增强 %d 次'%enc_time,"."*20)
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start:random_start + length]
                        Train_sample.append(sample)
                        # 从0开始，最终当samp_step为samp_train停止
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)

            # 从end_index后 抓取测试数据
            for h in range(number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples





    train_data=capture(path_train=d_path,filenames=filenames_train)
    test_data=capture(path_train=d_path1,filenames=filenames_test)

    return  train_data,test_data
if __name__=='__main__':
    path = r'F:\paderborn dataset\start_data\training_dataset\N09_M07_F10\all'
    data,data2=prepro(d_path=path)
    a=data['N09_M07_F10_K002_10.csv']
    a


