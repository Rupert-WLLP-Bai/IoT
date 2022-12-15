# Author: Norfloxaciner B
# Description: 预测
# Version: 1.0
# Date: 2022/12/15

import numpy as np
import pandas as pd
# 使用tensorflow-gpu进行时间序列预测
import tensorflow as tf
from matplotlib import pyplot as plt
import multiprocessing


def read_data():
    """读取数据
    """
    data = pd.read_csv("../data/data_datetime.csv", dtype={
        'Datetime': 'str',
        'Global_active_power': 'float',
        'Global_reactive_power': 'float',
        'Voltage': 'float',
        'Global_intensity': 'float',
        'Sub_metering_1': 'float',
        'Sub_metering_2': 'float',
        'Sub_metering_3': 'float'
    })
    print("len(df) = {}".format(len(data)))
    # 测试数据 前4320行
    data = data.head(14400).copy()

    # 将第一列的数据转换为时间格式
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    # 将后七列的数据转换为浮点数类型
    for column in data.columns[1:]:
        data[column] = pd.to_numeric(data[column])

    # 输出时间是否已经排序
    if data['Datetime'].is_monotonic_increasing:
        print("Datetime is monotonic increasing")
    else:
        print("Datetime is not monotonic increasing")
        raise Exception("Datetime is not monotonic increasing")
    return data


# 使用tensorflow-gpu进行时间序列预测
def predict_data(dataframe, column_name):
    # 训练集0.8 测试集0.2
    train_size = int(len(dataframe) * 0.8)
    test_size = len(dataframe) - train_size
    train, test = dataframe.iloc[0:train_size], dataframe.iloc[train_size:len(dataframe)]
    print("len(train) = {}".format(len(train)))
    print("len(test) = {}".format(len(test)))
    # 选择一个指标进行预测
    train = train[column_name]
    test = test[column_name]
    # 将数据转换为numpy数组
    train = train.values
    test = test.values
    # 使用tensorflow-gpu进行时间序列预测
    # 将数据转换为tensorflow的数据格式
    train_set = tf.data.Dataset.from_tensor_slices(train)
    test_set = tf.data.Dataset.from_tensor_slices(test)
    # 设置窗口大小
    window_size = 60
    # 设置批次大小
    batch_size = 32
    # 设置预测步长
    shuffle_buffer_size = 1000
    # 设置预测步长
    predict_step = 1
    predict_window_size = window_size + predict_step
    # 设置训练集
    train_set = train_set.window(window_size + predict_step, shift=1, drop_remainder=True)
    train_set = train_set.flat_map(lambda window: window.batch(predict_window_size))
    train_set = train_set.shuffle(shuffle_buffer_size).map(
        lambda window: (window[:-predict_step], window[-predict_step:]))
    train_set = train_set.batch(batch_size).prefetch(1)
    # 设置测试集
    test_set = test_set.window(window_size + predict_step, shift=1, drop_remainder=True)
    test_set = test_set.flat_map(lambda window: window.batch(predict_window_size))
    test_set = test_set.map(lambda window: (window[:-predict_step], window[-predict_step:]))
    test_set = test_set.batch(batch_size).prefetch(1)
    # 设置模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, input_shape=[window_size], activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    # 设置损失函数
    loss = tf.keras.losses.Huber()
    # 设置优化器
    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    # 设置评估指标
    metrics = ["mae"]
    # 编译模型
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # 训练模型
    history = model.fit(train_set, epochs=200)
    # 输出训练结果的mae和loss
    print("mae = {}".format(history.history['mae'][-1]))
    print("loss = {}".format(history.history['loss'][-1]))
    # 预测
    predict_result = model.predict(test_set)
    # 将预测结果转换为一维数组
    predict_result = predict_result.reshape(-1)
    # 绘制损失函数
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.legend(loc='upper left')
    plt.title("Loss {}".format(column_name))
    plt.savefig("../data/output/predict/Loss_{}.png".format(column_name))
    plt.show()
    # 绘制预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(test, label='Actual')
    plt.plot(predict_result, label='Predict')
    plt.legend(loc='upper left')
    plt.title("Predict {}".format(column_name))
    plt.savefig("../data/output/predict/Predict_{}.png".format(column_name))
    plt.show()
    # 保存结果到文件
    # 文件路径"../data/output/predict/{column}.csv"


if __name__ == "__main__":
    df = read_data()
    # 检查gpu是否可用
    print("tf.test.is_gpu_available() = {}".format(tf.test.is_gpu_available()))
    column_names = df.columns[1:]
    for column_name in column_names:
        predict_data(df, column_name)
