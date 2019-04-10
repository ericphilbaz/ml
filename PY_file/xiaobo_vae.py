# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from keras.layers import *
from keras import backend as K
from keras.models import Model
import imageio,os
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model
import preprocess
import matplotlib.pyplot as plt


# batch_size=100
latent_dim=100
# epochs=5
num_classes=7
img_dim=64
filters=16
intermediate_dim=1000


batch_size =128 #
epochs = 5
# num_classes =7   # 类别
length =4096   # 信号长度  16384=128*128
BatchNorm = True  # 是否批量归一化
number = 1000  # 每类样本的数量  1000:0.32,  10000:0.68,  20000:0.7105   30000:81
normal = True  # 是否标准化
rate = [0.8, 0.1, 0.1]  # 训练集验证集测试集划分比例,之和为1
#
path_training = r"/home/data/Yang_wanli/paderborn_dataset/all"
path_testing = r"/home/data/Yang_wanli/paderborn_dataset/all"
x_train, y_train_, x_valid, y_valid, x_test, y_test_=preprocess.prepro(d_path=path_training,d_path1=path_testing, length=length,number=number,normal=normal,rate=rate,enc=True, enc_step=28)

print(x_train.shape)
print(x_test.shape)
print(x_valid.shape)
print(y_train_)
print('*'*50)
x_train = x_train.reshape((-1, img_dim, img_dim, 1))
x_valid = x_valid.reshape((-1, img_dim, img_dim, 1))
x_test = x_test.reshape((-1, img_dim, img_dim, 1))
print(x_train.shape)
print(x_test.shape)
print(x_valid.shape)


# 搭建模型
x = Input(shape=(img_dim, img_dim, 1))
h = x
print("==========开始训练===========")
filters *= 2
for i in range(2):
    filters *= 2
    h = Conv2D(filters=filters,kernel_size=3,strides=2,padding='same')(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')(h)
    h = LeakyReLU(0.2)(h)

h4=h
h = LeakyReLU(0.2)(h4)
h_shape = K.int_shape(h)[1:]
h = Flatten()(h)
z_mean = Dense(latent_dim)(h) # p(z|x)的均值
z_log_var = Dense(latent_dim)(h) # p(z|x)的方差
encoder = Model(x, z_mean) # 通常认为z_mean就是所需的隐变量编码
z = Input(shape=(latent_dim,))
h = z
h = Dense(np.prod(h_shape))(h)
h = Reshape(h_shape)(h)

for i in range(2):
    h = Conv2DTranspose(filters=filters,kernel_size=3,strides=1,padding='same')(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2DTranspose(filters=filters,kernel_size=3,strides=2,padding='same')(h)
    h = LeakyReLU(0.2)(h)
    filters //= 2
x_recon = Conv2DTranspose(filters=1,kernel_size=3,activation='sigmoid',padding='same')(h)

decoder = Model(z, x_recon) # 解码器
generator = decoder

z = Input(shape=(latent_dim,))
y = Dense(intermediate_dim, activation='relu')(z)
y = Dense(num_classes, activation='softmax')(y)
classfier = Model(z, y) # 隐变量分类器

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
x_recon = decoder(z)
y = classfier(z)

class Gaussian(Layer):
    """这是个简单的层，定义q(z|y)中的均值参数，每个类别配一个均值。
    然后输出“z - 均值”，为后面计算loss准备。
    """
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)
    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',shape=(self.num_classes, latent_dim),initializer='zeros')
    def call(self, inputs):
        z = inputs # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z - K.expand_dims(self.mean, 0)
    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])

gaussian = Gaussian(num_classes)
z_prior_mean = gaussian(z)


# 建立模型
vae = Model(x, [x_recon, z_prior_mean, y])

# 下面一大通都是为了定义loss
z_mean = K.expand_dims(z_mean, 1)
z_log_var = K.expand_dims(z_log_var, 1)

lamb = 1.5 # 原来是2.5 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
xent_loss = 0.5 * K.mean((x - x_recon)**2, 0)
kl_loss = - 0.5 * (z_log_var - K.square(z_prior_mean))
kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)
cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
vae_loss = lamb * K.sum(xent_loss) + K.sum(kl_loss) + K.sum(cat_loss)


vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
vae.fit(x_train,shuffle=True,epochs=epochs,batch_size=batch_size,validation_data=(x_test, None),verbose=2)
plot_model(model=vae, to_file='vae_keras_cluster.png', show_shapes=True)


means = K.eval(gaussian.mean)
x_train_encoded = encoder.predict(x_train)
y_train_pred = classfier.predict(x_train_encoded).argmax(axis=1)
x_test_encoded = encoder.predict(x_test)
y_test_pred = classfier.predict(x_test_encoded).argmax(axis=1)


# def cluster_sample(path, category=0):
#     """观察被模型聚为同一类的样本
#     """
#     n = 8
#     figure = np.zeros((img_dim * n, img_dim * n))
#     idxs = np.where(y_train_pred == category)[0]
#     for i in range(n):
#         for j in range(n):
#             ## np.random.choice(a=5, size=3, replace=False, p=None)参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
#             digit = x_train[np.random.choice(idxs)]
#             digit = digit.reshape((img_dim, img_dim))
#             figure[i * img_dim: (i + 1) * img_dim,
#             j * img_dim: (j + 1) * img_dim] = digit
#     imageio.imwrite(path, figure * 255)


def random_sample(path, category=0, std=1):
    """按照聚类结果进行条件随机生成
    """
    n = 8
    figure = np.zeros((img_dim * n, img_dim * n))
    for i in range(n):
        for j in range(n):
            noise_shape = (1, latent_dim)
            #重参数
            z_sample = np.array(np.random.randn(*noise_shape)) * std + means[category]
            #通过输入z_sample来预测出x_recon
            print('***z_sample***')
            print(z_sample)
            print(z_sample.shape)
            x_recon = generator.predict(z_sample)
            digit = x_recon[0].reshape((img_dim, img_dim))
            figure[i * img_dim: (i + 1) * img_dim,j * img_dim: (j + 1) * img_dim] = digit
    imageio.imwrite(path, figure*255)


if not os.path.exists('samples'):
    os.mkdir('samples')

for i in range(7):
    # cluster_sample(u'samples/聚类类别_%s.png' % i, i)
    random_sample(u'samples/类别采样_%s.png' % i, i)

right = 0.
for i in range(7):
    print('%'*20)
    print(y_train_)
    print('%'*20)
    _ = np.bincount(y_train_[y_train_pred == i])
    right += _.max()

print('train acc: %s' % (right / len(y_train_)))

right = 0.
for i in range(7):
    _ = np.bincount(y_test_[y_test_pred == i])
    right += _.max()

print('test acc: %s' % (right / len(y_test_)))