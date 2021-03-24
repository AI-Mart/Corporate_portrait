import csv
import random
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid

paddle.enable_static()

def reader_creater(filename):
    def reader():
        with open(filename, 'r', encoding='gbk') as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if csv_reader.line_num == 1:
                    continue
                yield [float(each) for each in item[1:-1]], int(item[-1])
    return reader

BUF_SIZE = 12800
BATCH_SIZE = 128

train_reader = paddle.batch(
    paddle.reader.shuffle(reader_creater('train_data_preprocessed_2.csv'),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)
test_reader = paddle.batch(
    paddle.reader.shuffle(reader_creater('test_data_preprocessed_2.csv'),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)

# 定义多层感知器
def multilayer_perception(input):
    # 第一个全连接层，激活函数为 ReLU
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
    # 第二个全连接层，激活函数为 ReLU
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
    # 第三个全连接层，激活函数为 ReLU
    hidden3 = fluid.layers.fc(input=hidden2, size=100, act='relu')
    # 以 softmax 为激活函数的全连接输出层，输出层的大小为种类的个数 2
    prediction = fluid.layers.fc(input=hidden3, size=2, act='softmax')
    return prediction

input = fluid.layers.data(name='input', shape=[27], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
predict = multilayer_perception(input)

# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict, label=label)  # 交叉熵
avg_cost = fluid.layers.mean(cost)  # 计算 cost 中所有元素的平均值
acc = fluid.layers.accuracy(input=predict, label=label)  # 使用输入和标签计算准确率

# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)
print('完成')


# 定义使用 CPU 还是 GPU，使用 CPU 时 use_cuda = False，使用 GPU 时 use_cuda = True
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# 创建执行器，初始化参数
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(feed_list=[input, label], place=place)


def draw_train_process(title, iters, costs, accs, label_cost, lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel('iter', fontsize=20)
    plt.ylabel('cost/acc', fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()


EPOCH_NUM = 20
model_save_dir = 'zombie_enterprise.inference.model'

all_test_iter = 0
all_test_iters = []
all_test_costs = []
all_test_accs = []

for pass_id in range(EPOCH_NUM):
    # 开始训练
    for batch_id, data in enumerate(train_reader()):  # 遍历 train_reader 的迭代器，并为数据加上索引 batch_id
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                        feed=feeder.feed(data),  # 喂入一个 batch 的数据
                                        fetch_list=[avg_cost, acc])  # fetch 均方误差和准确率

        # 每 50 次 batch 打印一次训练
        if batch_id % 50 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 每趟进行一次测试
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,  # 执行测试程序
                                      feed=feeder.feed(data),  # 喂入数据
                                      fetch_list=[avg_cost, acc])  # fetch 误差、准确率
        all_test_iter += 1
        all_test_iters.append(all_test_iter)
        all_test_costs.append(test_cost[0])  # 记录每个 batch 的误差
        all_test_accs.append(test_acc[0])  # 记录每个 batch 的准确率

    # 求测试结果的平均值
    test_cost = (sum(all_test_costs) / len(all_test_costs))  # 计算误差平均值（误差和/误差的个数）
    test_acc = (sum(all_test_accs) / len(all_test_accs))  # 计算准确率平均值（ 准确率的和/准确率的个数）
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))

# 保存模型
# 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to {}'.format(model_save_dir))
fluid.io.save_inference_model(model_save_dir,
                              ['input'],
                              [predict],
                              exe)
print('训练模型保存完成！')


draw_train_process('test', all_test_iters, all_test_costs, all_test_accs, 'test cost', 'test acc')