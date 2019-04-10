from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import pywt
import pandas as pd
#rate  训练集/验证集/测试集比例
def prepro(d_path,length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
    """
    # 获得该文件夹下所有.mat文件名
    print('train_path:', d_path)
    # print('test_path:', d_path1)
#     filenames = os.listdir(d_path)
#     filenames_test = os.listdir(d_path1)
    # print(filenames)
    # print("#"*30)

    
    filenames_train = os.listdir(d_path)
    # filenames_out_test = os.listdir(d_path1)
    def capture(path,filenames):
        """读取csv文件，返回数据
        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(path, i)
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

    def slice_enc(data,d_path_add,slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.
        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        # keys=os.listdir(d_path_add)
        # print(keys)
        Train_Samples = {}
        Test_Samples = {}
        #从end_index前抓取训练集数据
        for i in keys:
            slice_data = data[i]
            all_lenght = len(slice_data)
            # print(all_lenght)
            end_index = int(all_lenght * (1 - slice_rate))
            # print(end_index)
            samp_train = int(number * (1 - slice_rate))  # 700
            # print(samp_train)
            # Train_Sample_pd=pd.DataFrame()
            # Test_Sample_pd=pd.DataFrame()
            ##采用list来append,提高效率
            Train_sample_a = []
            Train_sample_d = []

            Test_Sample_a = []
            Test_Sample_d = []

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
                        sample_a = sample['a'].tolist()
                        # print(len(sample_a))
                        Train_sample_a.append(sample_a)
                        sample_d = sample['d'].tolist()
                        Train_sample_d.append(sample_d)
                        #从0开始，最终当samp_step为samp_train停止
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]

                    sample_a=sample['a'].tolist()
                    Train_sample_a.append(sample_a)
                    sample_d = sample['d'].tolist()
                    Train_sample_d.append(sample_d)

            # 从end_index后 抓取测试数据  
            for h in range(number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                sample_a = sample['a'].tolist()
                Test_Sample_a.append(sample_a)
                sample_d = sample['d'].tolist()
                Test_Sample_d.append(sample_d)
            # Train_Sample_pd = Train_Sample_pd.append(Train_sample)
            # Test_Sample_pd = Test_Sample_pd.append(Test_Sample)
            # Train_Sample_pd['a']=Train_sample_a
            # Train_Sample_pd['d']=Train_sample_d
            # Test_Sample_pd['a']=Test_Sample_a
            # Test_Sample_pd['d']=Test_Sample_d
            # Train_Samples.update(dict(zip([i], [Train_Sample_pd])))
            # Test_Samples.update(dict(zip([i], [Test_Sample_pd])))
            # Train_Samples[i]=Train_Sample_pd.tolist()
            # print(len(Train_Sample_pd))
            # Test_Samples[i]=Test_Sample_pd.tolist()
            Train_Samples[i]=[Train_sample_a,Train_sample_d]
            Test_Samples[i]=[Test_Sample_a,Test_Sample_d]

        # print(len(Train_Samples))

            # Train_Samples[i] = Train_sample
            # Test_Samples[i] = Test_Sample

        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签  将两组数据拆分便于后面归一化
    def add_labels(data,filenames):
        Y = []
        X_1=[]
        X_2=[]
        label = 0
        for i in filenames:
            X_a = []
            X_d = []
            x_a = data[i][0]
            x_d = data[i][1]
            # print(type(x_a))
            X_a+=x_a
            X_d+=x_d
            X_1+=X_a
            X_2+=X_d
            lenx = len(x_a)
            # print(lenx)
            # print(lenx)
            Y += [label] * lenx  #定义标签的长度  X为train 的标签   Y为test的标签
            # print([label])
            # print(Y)
            label += 1
        # print(X.shape)
        return X_1,X_2, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        # print(Train_Y)
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        # print(len(Train_Y))
        # print("*"*50)
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        # print(Train_Y)
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        # print(Test_X)
        # print(len(Test_X))
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y, n):   #  默认n=3

        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=n, test_size=test_size)    #分层采样
        # print(ss)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]

            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test
    data = capture(path=d_path,filenames=filenames_train)
    # data1 = capture(path=d_path1,filenames=filenames_out_test)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data,d_path_add=d_path)
    Train_X_1, Train_X_2,Train_Y = add_labels(data=train,filenames=filenames_train)
    Test_X_1,Test_X_2, Test_Y = add_labels(data=test, filenames=filenames_train)
    # 为第二个训练集\测试集制作标签，返回X，Y
    # 为训练集Y/测试集One-hot标签
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    print("\\"*20,'train_one-hot',"\\"*20)
    #验证集  vae暂时不onehot
    # print(Train_X)
    print("\\"*20,'test_one-hot',"\\"*20)
    # 训练数据/测试数据 是否标准化
    if normal:
        print("\\"*20,'train_scalar-stand',"\\"*20)
        Train_X_1,Test_X_1=scalar_stand(Train_X_1,Test_X_1)
        Train_X_2,Test_X_2=scalar_stand(Train_X_2,Test_X_2)

        Train_X=[Train_X_1,Train_X_2]
        Test_X=[Test_X_1,Test_X_2]
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)
        print("\\"*20,'test_scalar-stand',"\\"*20)
        # Train_X_OUT, Test_X_OUT= scalar_stand(Train_X_OUT, Test_X_OUT)
    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X = [Train_X_1, Train_X_2]
        Test_X = [Test_X_1, Test_X_2]
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)
        # Train_X_OUT = np.asarray(Train_X_OUT)
    # 将测试集切分为验证集合和测试集.
    print(Train_X.shape)
    print(Train_Y.shape)
    print(Test_X.shape)
    print(Test_Y.shape)

    # Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X,Test_Y,n=10)
    # Valid_X1, Valid_Y1, Test_X, Test_Y = valid_test_slice(Test_X_OUT, Test_Y_OUT,n=10)
    print("*"*30,'数据加载完毕',"*"*30)
    return Train_X, Train_Y,Test_X, Test_Y
if __name__=='__main__':
    #######这是为了满足跨数据测试需要,可以设置其他路径
    path_training = r"F:\paderborn dataset\start_data\training_dataset\N09_M07_F10\all"
    BatchNorm = True  # 是否批量归一化
    number = 1000  # 每类样本的数量  1000:0.32,  10000:0.68,  20000:0.7105   30000:81
    length = 8100  # 信号长度  16384=128*128
    normal = True  # 是否标准化
    enc = True  # 256000/6400=40  小于3000，所以要做数据增强
    enc_step = 128
    rate = [0.8, 0, 0.2]  # 训练集验证集测试集划分比例,之和为1
    x_train1, y_train1,x_test1, y_test1 = prepro(d_path=path_training,length=length,number=number, normal=normal,
                                                                                 rate=rate, enc=enc, enc_step=enc_step)
    np.save('../AA_data/train_x.npy', x_train1)
    np.save('train_y.npy', y_train1)
    np.save('test_x.npy',x_test1)
    np.save('test_y.npy', y_test1)
    print("*"*30,'数据保存完毕',"*"*30)
