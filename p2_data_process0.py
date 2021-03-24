import csv
import random
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid


from p1_data_count import get_digit_num


year_report_num = get_digit_num('year_report.csv')
money_num = get_digit_num('money.csv')

###把同一企业，某一个属性存在多条的数据进行汇总，然后取均值
def preprocess_0(filename, num):
    # 对 year_report_train.csv 和 money_train.csv 中的数据进行预处理
    # 舍弃年份信息，对其他数据取均值
    # 如果数据类型是小数，保留 num 个小数位

    with open(filename, 'r', encoding='gbk') as f:

        reader = csv.reader(f)
        companies = {}
        result = []

        # 将所有数据按公司 ID 分开存放
        for line in reader:
            if reader.line_num == 1:
                title = ','.join([line[0]] + line[2:])
                result.append(title)
            else:
                if line[0] not in companies:
                    companies[line[0]] = [line[2:]]
                else:
                    companies[line[0]].append(line[2:])

        # 对每家公司，舍弃年份信息，对其他数据取均值
        for id, data in companies.items():
            temp = [id]
            for i in range(len(data[0])):
                count, sum, is_float = 0, 0, False
                for line in data:
                    if line[i] != 'NA':
                        count += 1
                        if '.' in line[i]:
                            is_float = True
                            sum += float(line[i])
                        else:
                            sum += int(line[i])
                if is_float:
                    temp.append(str(round(sum / len(data), num)))  # 小数数据，保留一位小数，与原数据保持一致
                else:
                    temp.append(str(round(sum / len(data))))  # 整数数据
            result.append(','.join(temp))

    with open(filename.split('.')[0] + '_preprocessed.csv', 'w', encoding='gbk') as f:
        f.write('\n'.join(result))

if __name__=="__main__":
    preprocess_0('year_report.csv', year_report_num)
    preprocess_0('money.csv', money_num)