import csv
import random
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid



def preprocess_2(filename):
    # 对 knowledge_train.csv 中的数据进行预处理，用相同的概率分布填补缺失值

    print('\n正在处理 {} 文件的数据...\n'.format(filename))

    with open(filename, 'r', encoding='gbk') as f:
        reader = csv.reader(f)
        data = list(reader)
        title = ','.join(data[0])

        ones = [0, 0, 0]
        alls = [0, 0, 0]
        result = [title]

        # 遍历一遍，计算各属性中 1 的占比（概率分布）
        for line in data[1:]:
            for i in range(3):
                if line[i + 8] != 'NA':
                    alls[i] += 1
                    if line[i + 8] == '1':
                        ones[i] += 1

        title = ['专利', '商标', '著作权']
        rate = []
        for i in range(3):
            rate.append(ones[i] / alls[i])
            print('[{}] 属性中 1 的占比为 {:.3f}%'.format(title[i], rate[i] * 100))

        # 用相同的概率分布填补缺失值
        # 具体为：获取一个 0-1 之间的随机数
        # 如果随机数大于该属性中 1 的占比，则填补为 1，否则填补为 0
        for line in data[1:]:
            for i in range(3):
                if line[i + 8] == 'NA':
                    if random.random() > rate[i]:
                        line[i + 8] = '1'
                    else:
                        line[i + 8] = '0'
            result.append(','.join(line))

    with open(filename.split('_preprocessed_1')[0] + '_preprocessed_2.csv', 'w', encoding='gbk') as f:
        f.write('\n'.join(result))

if __name__=="__main__":
    preprocess_2('train_data_preprocessed_1.csv')
    preprocess_2('test_data_preprocessed_1.csv')