from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同

def prepro(d_path,d_path1,length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

    ```
    import preprocess.preprocess_nonoise as pre

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = pre.prepro(d_path=path,
                                                                    length=864,
                                                                    number=1000,
                                                                    normal=False,
                                                                    rate=[0.5, 0.25, 0.25],
                                                                    enc=True,
                                                                    enc_step=28)
    ```
    """
    # 获得该文件夹下所有.mat文件名
    print('train:', d_path)
    print('test:', d_path1)
    filenames = os.listdir(d_path)
    filenames_test = os.listdir(d_path1)
    # print(filenames)
    # print("#"*30)

    def capture(path_train):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(path_train, i)
            # print(i)
            # print(file_path)

            file = np.loadtxt(file_path,skiprows=1)
            # files=file
            # files = dict(zip([i], [file]))
            files.update(dict(zip([i], [file])))
        return files

        #     file = loadmat(file_path)  # 在mat文件中寻找DE
        #     file_keys = file.keys()
        #     for key in file_keys:
        #         if 'cfs_1_1' in key:
        #             files[i] = file[key].ravel()
        #
        # return files

    def capture1(path_test):
        files_1 = {}
        for i in filenames_test:
            # 文件路径
            file_path_1= os.path.join(path_test, i)
        #     # print(i)
        #     # print(file_path)

            file_1= np.loadtxt(file_path_1,skiprows=1)
            # files=file
            # files = dict(zip([i], [file]))
            files_1.update(dict(zip([i], [file_1])))
        return files_1



        #     file = loadmat(file_path_1)  #在mat文件中寻找DE
        #     file_keys = file.keys()
        #     for key in file_keys:
        #         if 'cfs_1_0' in key:
        #             files_1[i] = file[key].ravel()
        # return files_1

    def slice_enc(data,d_path_add,slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        # keys = data.keys()
        keys=os.listdir(d_path_add)
        # print(keys)
        Train_Samples = {}
        Test_Samples = {}
        # print('*'*50)
        # print(data)
        # print('*'*50)
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
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
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

            # 抓取测试数据
            for h in range(number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            # print(lenx)
            Y += [label] * lenx  #定义标签的长度  X为train 的标签   Y为test的标签
            # print([label])
            # print(Y)
            label += 1
        return X, Y

    def add_labels_test(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames_test:
            x = train_test[i]
            X += x
            lenx = len(x)
            # print(lenx)
            Y += [label] * lenx  #定义标签的长度  X为train 的标签   Y为test的标签
            # print([label])
            # print(Y)
            label += 1
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        # print(Train_Y)
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        # print(len(Train_Y))
        # print("*"*50)
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        # Train_Y = Encoder.transform(Train_Y).toarray()
        # print(Train_Y)
        # Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        # print(Test_X)
        print("+"*50)
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
            # print(len(train_index))
            # print(test_index)
            # print(len(test_index))
            # print("+" * 50)
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = capture(path_train=d_path)
    data1 = capture1(path_test=d_path1)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data,d_path_add=d_path)
    train_OUT, test_OUT= slice_enc(data1,d_path_add=d_path1)      #######这是为了满足跨数据测试需要
    # 为第一个训练集\测试集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    Test_X, Test_Y= add_labels(test)
    # 为第二个训练集\测试集制作标签，返回X，Y                        #######这是为了满足跨数据测试需要
    Train_X_OUT, Train_Y_OUT= add_labels_test(train_OUT)
    Test_X_OUT, Test_Y_OUT = add_labels_test(test_OUT)
    # 为训练集Y/测试集One-hot标签   vae暂时不onehot
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    #验证集  vae暂时不onehot
    Train_Y_OUT, Test_Y_OUT = one_hot(Train_Y_OUT, Test_Y_OUT)
    # 训练数据/测试数据 是否标准化
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
        Train_X_OUT, Test_X_OUT= scalar_stand(Train_X_OUT, Test_X_OUT)
    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)
        # Train_X_OUT = np.asarray(Train_X_OUT)
        Test_X_OUT = np.asarray(Test_X_OUT)
###
    # 将测试集切分为验证集合和测试集.
    Valid_X, Valid_Y, Test_X1, Test_Y1 = valid_test_slice(Test_X,Test_Y,n=10)
    Valid_X1, Valid_Y1, Test_X, Test_Y = valid_test_slice(Test_X_OUT, Test_Y_OUT,n=10)
    print('数据加载完毕')
    print('￥' * 20)
    Train_Y=Train_Y.reshape(len(Train_Y),)
    Test_Y=Test_Y.reshape(len(Test_Y),)
    Valid_Y=Valid_Y.reshape(len(Valid_Y),)
    print(Train_Y)
    print('￥' * 20)
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
