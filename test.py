import pandas as pd
import numpy as np
import math
import csv

# np.save('weight.npy', w)
# 读入测试数据test.csv
testdata = pd.read_csv('./data/test.csv', header = None, encoding = 'big5')
# 丢弃前两列，需要的是从第3列开始的数据
test_data = testdata.iloc[:, 2:].copy()
# 把降雨为NR字符变成数字0
test_data[test_data == 'NR'] = 0.2
# 将dataframe变成numpy数组
test_data = test_data.to_numpy()
# 将test数据也变成 240 个维度为 18 * 9 + 1 的数据。
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
# print(test_x)
test_x = (test_x-test_x.mean())/test_x.std()
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
# print(ans_y)


with open('./data/answer1.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)



