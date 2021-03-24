p1_data_count.py   
统计数据集中不完整的记录占比

p2_data_process0.py 
把同一企业，某一个属性存在多条的数据进行汇总，然后取均值, 舍弃年份信息，对其他数据取均值, 如果数据类型是小数，保留 num 个小数位,
生成money_preprocessed.csv,year_report_preprocessed.csv

p3_data_process1.py
合并四个文件的数据,base.csv,knowledge.csv,money_preprocessed.csv,year_report_preprocessed.csv
生成all_data.csv

p4_train_test_data0.py 
按比例分开训练集和测试集 
生成train_data.csv和test_data.csv

p5_train_test_data1.py
对 base_train.csv 中的数据进行预处理
注册时间、行业、区域、企业类型、控制人类型：用相同的概率分布填补缺失值，并替换成数字
注册资本、控制人持股：用均值填补缺失值
舍弃无关属性控制人 ID
舍弃没有标签的数据 
生成train_data_preprocessed_1.csv和test_data_preprocessed_1.csv

p6_train_test_data2.py
对 knowledge_train.csv 中的数据进行预处理，用相同的概率分布填补缺失值
生成train_data_preprocessed_2.csv和test_data_preprocessed_2.csv

p7_model_train.py
模型训练
生成zombie_enterprise.inference.model

p8_model_predict.py
模型预测