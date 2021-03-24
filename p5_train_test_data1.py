import csv
import random
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid


zcsj_values = ['2000', '2001', '2002', '2003', '2004',
               '2005', '2006', '2007', '2008', '2009',
               '2010', '2011', '2012', '2013', '2014']
hy_values = ['商业服务业', '零售业', '工业', '服务业', '社区服务', '交通运输业']
qy_values = ['江西', '湖北', '广西', '湖南', '福建', '广东', '山东']
qylx_values = ['有限责任公司', '集体所有制企业', '合伙企业', '农民专业合作社', '股份有限公司']
kzrlx_values = ['自然人', '企业法人']


def preprocess_1(filename):
    # 对 base_train.csv 中的数据进行预处理
    # 注册时间、行业、区域、企业类型、控制人类型：用相同的概率分布填补缺失值，并替换成数字
    # 注册资本、控制人持股：用均值填补缺失值
    # 舍弃无关属性控制人 ID
    # 舍弃没有标签的数据

    print('\n正在处理 {} 文件的数据...\n'.format(filename))

    with open(filename, 'r', encoding='gbk') as f:
        reader = csv.reader(f)
        data = list(reader)
        title = ','.join(data[0][:7] + data[0][8:])
        result = [title]

        zcsj, hy, qy, qylx, kzrlx = {}, {}, {}, {}, {}
        zczb_count, zczb_all, kzrcg_count, kzrcg_all = 0, 0, 0, 0

        # 遍历一遍，计算各属性的概率分布/均值
        for line in data[1:]:
            if line[1] != 'NA':
                if line[1] in zcsj:
                    zcsj[line[1]] += 1
                else:
                    zcsj[line[1]] = 1
            if line[2] != 'NA':
                zczb_count += 1
                zczb_all += float(line[2])
            if line[3] != 'NA':
                if line[3] in hy:
                    hy[line[3]] += 1
                else:
                    hy[line[3]] = 1
            if line[4] != 'NA':
                if line[4] in qy:
                    qy[line[4]] += 1
                else:
                    qy[line[4]] = 1
            if line[5] != 'NA':
                if line[5] in qylx:
                    qylx[line[5]] += 1
                else:
                    qylx[line[5]] = 1
            if line[6] != 'NA':
                if line[6] in kzrlx:
                    kzrlx[line[6]] += 1
                else:
                    kzrlx[line[6]] = 1
            if line[8] != 'NA':
                kzrcg_count += 1
                kzrcg_all += float(line[8])

        def gen_random_value(d):
            # 根据概率分布获取随机值
            value = random.uniform(0, sum(d.values()))
            value_sum = 0
            for key in d:
                value_sum += d[key]
                if value <= value_sum:
                    return key

        def test_random(d):
            # 查看 gen_random_value 生成的随机数的概率分布
            value_sum = sum(d.values())
            print('\n原始概率分布=>')
            for key, value in d.items():
                print('{}: {}'.format(key, round(value / value_sum, 5)))

            # print('\n'.join([':'.join(items) for items in temp_d.items()]))
            print('\n随机数的概率分布=>')
            result = []
            for _ in range(100000):
                value = gen_random_value(d)
                result.append(value)
            for key, value in Counter(result).items():
                print('{}: {}'.format(key, value / 100000))

        print('=== 注册时间 ===')
        test_random(zcsj)
        print('\n\n\n=== 行业 ===')
        test_random(hy)
        print('\n\n\n=== 区域 ===')
        test_random(qy)
        print('\n\n\n=== 企业类型 ===')
        test_random(qylx)
        print('\n\n\n=== 控制人类型 ===')
        test_random(kzrlx)

        # 注册时间、行业、区域、企业类型、控制人类型：用相同的概率分布填补缺失值
        # 注册资本、控制人持股：用均值填补缺失值
        for line in data[1:]:
            # 舍弃没有标签的数据
            if line[-1] == 'NA':
                continue
            # 统计发现，数据里的概率分布是均匀分布，故随机填充一属性值即可
            if line[1] == 'NA':
                line[1] = random.sample(list(zcsj), 1)[0]
            if line[2] == 'NA':
                line[2] = str(round(zczb_all / zczb_count, 2))
            if line[3] == 'NA':
                line[3] = random.sample(list(hy), 1)[0]
            if line[4] == 'NA':
                line[4] = random.sample(list(qy), 1)[0]
            if line[5] == 'NA':
                line[5] = random.sample(list(qylx), 1)[0]
            if line[6] == 'NA':
                line[6] = random.sample(list(kzrlx), 1)[0]
            if line[8] == 'NA':
                line[8] = str(round(kzrcg_all / kzrcg_count, 2))
            line[1] = str(zcsj_values.index(line[1]))
            line[3] = str(hy_values.index(line[3]))
            line[4] = str(qy_values.index(line[4]))
            line[5] = str(qylx_values.index(line[5]))
            line[6] = str(kzrlx_values.index(line[6]))
            result.append(','.join(line[:7] + line[8:]))

    with open(filename.split('.')[0] + '_preprocessed_1.csv', 'w', encoding='gbk') as f:
        f.write('\n'.join(result))

if __name__=="__main__":
    preprocess_1('train_data.csv')
    preprocess_1('test_data.csv')