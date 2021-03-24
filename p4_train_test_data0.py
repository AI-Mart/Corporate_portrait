import csv
import random
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid



def separate_train_and_test(rate):
    # 按比例分开训练集和测试集

    with open('all_data.csv', 'r', encoding='gbk') as f:
        reader = csv.reader(f)
        all_data = list(reader)

    title = all_data[0]
    all_data = all_data[1:]
    random.shuffle(all_data)
    train_len = round(len(all_data) * 0.8)
    train_text = ','.join(title) + '\n' + '\n'.join([','.join(line) for line in all_data[:train_len]])
    test_text = ','.join(title) + '\n' + '\n'.join([','.join(line) for line in all_data[train_len:]])

    with open('train_data.csv', 'w', encoding='gbk') as f:
        f.write(train_text)

    with open('test_data.csv', 'w', encoding='gbk') as f:
        f.write(test_text)

if __name__=="__main__":
    separate_train_and_test(0.8)