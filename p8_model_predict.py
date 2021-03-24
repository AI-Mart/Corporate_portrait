import csv
import random
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid

model_save_dir = 'zombie_enterprise.inference.model'
paddle.enable_static()

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

input = fluid.layers.data(name='input', shape=[27], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
# 定义输入数据维度
feeder = fluid.DataFeeder(feed_list=[input, label], place=place)

# 创建预测用的 Executor
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()

def predict(infer_path, num=10):
    with fluid.scope_guard(inference_scope):
        # 从指定目录中加载 inference model
        [inference_program,  # 预测用的 program
         feed_target_names,  # 是一个 str 列表，它包含需要在推理 Program 中提供数据的变量的名称
         fetch_targets] = fluid.io.load_inference_model(model_save_dir,  # fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果
                                                        infer_exe)  # infer_exe: 运行 inference model 的 executor

    with open('test_data_preprocessed_2.csv', 'r', encoding='gbk') as f:
        reader = csv.reader(f)
        data = list(reader)
        title = data[0]
        data = data[1:]

    sample_data = random.sample(data, num)
    for i in range(len(sample_data)):
        line = sample_data[i]
        for j in range(1, len(line) - 1):
            line[j] = float(line[j])

        # 生成预测数据
        tensor = np.expand_dims(line[1:-1], axis=0).astype(np.float32)
        results = infer_exe.run(inference_program,  # 运行预测程序
                                feed={feed_target_names[0]: tensor},  # 喂入要预测的 data
                                fetch_list=fetch_targets)  # 得到推测结果

        labels = {
            0: '非僵尸企业',
            1: '僵尸企业'
        }
        print('=== 第 {} 个预测结果 =='.format(i + 1))
        for x in range(len(line) - 1):
            print('{}：{}'.format(title[x], line[x]))
        print('\n')
        print('=>真实结果：{}'.format(labels[int(line[-1])]))
        print('=>预测结果：{}'.format(labels[np.argmax(results[0])]))
        print('\n\n\n')

if __name__=="__main__":
    predict('zombie_enterprise.inference.model')