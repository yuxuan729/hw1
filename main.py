import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visdom
import torch

# vis = visdom.Visdom(env=u'hw1')

data = pd.read_csv('./data/train.csv', encoding='big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
# print(raw_data)

month_data = {}

for month in range(12):
    sample = np.zeros((18, 24*20))
    for day in range(20):
        sample[:, day*24:(day+1)*24] = raw_data[(day+month*20)*18:(day+1+month*20)*18, :]
    month_data[month] = sample
# print(month_data[0].shape)

train_x = np.empty((12,471,18,9))  # 12个月，每个月471组数据，（18,9）数据格式
train_y = np.empty((12,471,1))  # 12个月，每个月471组数据，（1）数据格式
for month in range(12):
    for index in range(471):
        train_x[month][index] = month_data[month][:, index:index+9]
        train_y[month][index] = month_data[month][9, index+9]
# print(train_x[11][0])
# print(train_y[11][0])

#  对输入数据进行归一化操作
train_x = (train_x-train_x.mean())/train_x.std()
# for i in range(12):
#     for j in range(471):
#         for k in range(18):
#             if train_x[i][j][k].std() != 0:
#                 train_x[i][j][k] = (train_x[i][j][k]-train_x[i][j][k].mean())/train_x[i][j][k].std()

data_x = train_x.reshape(-1, 18*9)  # 5652*162
data_y = train_y.reshape(471*12, -1)

train_x = data_x[:5000, :]  # 5000*162
train_y = data_y[:5000, :]  # 5000*162

valid_x = data_x[5000:, :]
valid_y = data_y[5000:, :]

w_dim = 18*9+1  # 将b替换为w0
train_x = np.concatenate((np.ones((int(train_x.shape[0]),1)),train_x),axis=1).astype(float)
valid_x = np.concatenate((np.ones((int(valid_x.shape[0]),1)),valid_x),axis=1).astype(float)
w = np.random.rand(w_dim,1,)
adagrad = np.zeros([w_dim, 1])



for t in range(1000):
    loss = np.sqrt(np.sum(np.power(np.dot(train_x, w) - train_y, 2)) / 5000)
    if t % 100 == 0:
        print(str(t) + ":" + str(loss))
    gradient = np.dot(train_x.transpose(), np.dot(train_x, w) - train_y)
    adagrad += gradient ** 2  # adagrad中，学习率要除以历史所有的梯度平方和  / np.sqrt(adagrad + eps)
    w = w - 10 * gradient/ np.sqrt(adagrad)


    # visdom 可视化
    # vis.line(X=[t], Y=[loss],
    #          win='loss',
    #          opts=dict(title='loss', xlable='epoch'),
    #          update='append')
np.save('weight.npy',w)
pre_y = np.dot(valid_x,w)
print(np.concatenate((pre_y,valid_y),axis=1))
