import csv
import random
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid


def merge():
    # 合并四个文件的数据

    def initial(filename):
        with open(filename, 'r', encoding='gbk') as f:
            reader = csv.reader(f)
            data = list(reader)
        title.extend(data[0])
        data = data[1:]
        for line in data:
            if line[0] == '6000000':
                # 数据异常，丢弃
                continue
            result[line[0]] = line

    def add_data(filename):
        with open(filename, 'r', encoding='gbk') as f:
            reader = csv.reader(f)
            for line in reader:
                if reader.line_num == 1:
                    title.extend(line[1:])
                    continue
                if line[0] in result:
                    result[line[0]].extend(line[1:])

    result = {}
    title = []

    initial('base.csv')
    add_data('knowledge.csv')
    add_data('money_preprocessed.csv')
    add_data('year_report_preprocessed.csv')

    result_text = '\n'.join([','.join(line[:9] + line[10:] + [line[9]]) for line in result.values()])
    title = ','.join(title[:9] + title[10:] + [title[9]])
    text = title + '\n' + result_text
    with open('all_data.csv', 'w', encoding='gbk') as f:
        f.write(text)

if __name__=="__main__":
    merge()