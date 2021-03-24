import csv
import random
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid



def count_na_line(filename):
    # 统计数据集中不完整的记录占比
    with open(filename, 'r', encoding='gbk') as f:
        csv_reader = csv.reader(f)
        all, fail = 0, 0
        for line in csv_reader:
            if csv_reader.line_num == 1:
                continue
            if 'NA' in line:
                fail += 1
            all += 1
    print('{} 文件里共有 {} 条记录，其中 {} 条记录不完整，约为 {:.3f}%'.format(
        filename, all, fail, fail / all * 100))


def get_digit_num(filename):
    with open(filename, 'r', encoding='gbk') as f:
        reader = csv.reader(f)
        num = 0
        for line in reader:
            for item in line:
                if '.' in item and 'E' not in item:
                    num = max(num, len(item.split('.')[1]))
    print('{} 文件中小数精确到 {} 位'.format(filename, num))
    return num



if __name__=="__main__":
    count_na_line('base.csv')
    count_na_line('knowledge.csv')
    count_na_line('year_report.csv')
    count_na_line('money.csv')

    get_digit_num('base.csv')
    get_digit_num('knowledge.csv')
    year_report_num = get_digit_num('year_report.csv')
    money_num = get_digit_num('money.csv')