# # Author: Norfloxaciner B
# Description: 画图
# Version: 1.0
# Date: 2022/12/14

import os
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing import Pool, Process, Manager
from time import sleep

# 根据时间画出每1小时的用电量
# 数据格式:
# Datetime,Global_active_power,Global_reactive_power,Voltage,Global_intensity,Sub_metering_1,Sub_metering_2,Sub_metering_3
# 时间,全局有功功率,全局无功功率,电压,全局电流强度,子电表1,子电表2,子电表3
# 数据起始时间:2006/12/16 17:24
# 数据结束时间:2008/12/13 21:38
# 数据间隔:1分钟
# 每24小时画出一个图
# 输出文件路径: ../data/output/{指标名称}
# 输出文件名: {起始时间}-{结束时间}.png
# 设置画图风格
# 设置图例和标题

# 记录进度的全局变量
global amount
global current
global df


def plot_all():
    # 读取数据
    global df
    df = pd.read_csv("../data/data_datetime.csv", dtype={
        'Datetime': 'str',
        'Global_active_power': 'float',
        'Global_reactive_power': 'float',
        'Voltage': 'float',
        'Global_intensity': 'float',
        'Sub_metering_1': 'float',
        'Sub_metering_2': 'float',
        'Sub_metering_3': 'float'
    })
    print("len(df) = {}".format(len(df)))
    # 测试数据 前4320行
    # df = df.head(4320)
    
    # 将第一列的数据转换为时间格式
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    # 将后七列的数据转换为浮点数类型
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column])

    # 输出时间是否已经排序
    if df['Datetime'].is_monotonic_increasing:
        print("Datetime is monotonic increasing")
    else:
        print("Datetime is not monotonic increasing")
        raise Exception("Datetime is not monotonic increasing")

    # 获取需要画图的数据总数，为总时间数乘总指标数
    global amount
    amount = len(df['Datetime']) * (len(df.columns) - 1)
    # 使用多线程画图
    # 定义一个列表，用于存放画图的参数
    # 定义一个列表，用于存放画图的线程
    # 定义一个列表，用于存放画图的线程池
    # 避免系统占用过多资源，设置最大线程数为4
    # 每一个指标用一个线程池画图
    # 每4小时画一个图，检查终止位置的索引是否超出数据总数
    # 画图的起始时间
    # 画图的终止时间
    # 画图的进度
    pool = []
    params = []
    max_thread = os.cpu_count()
    time_interval_minute = 1440
    for col in df.columns[1:]:
        pool.append(Pool(max_thread))
        # 处理2006/12/16 17:24:00到2006/12/16 23:59:00的数据
        # 方便后面的数据对齐整点
        first_start = pd.to_datetime("2006/12/16 17:24:00")
        first_end = pd.to_datetime("2006/12/16 23:59:00")
        params.append((col, first_start, first_end))
        # 从2006/12/17 00:00:00开始画图
        time_start = pd.to_datetime("2006/12/17 00:00:00")
        # 结束时间由df的最后一行的时间决定
        time_end = df['Datetime'][len(df['Datetime']) - 1]
        # print("time_end = {}".format(time_end))
        # 如果结束时间超出数据总数，则将结束时间设置为数据的最后一行
        while True:
            start_time = time_start
            end = time_start + pd.Timedelta(minutes=time_interval_minute)
            if end > df['Datetime'][len(df['Datetime']) - 1]:
                end = df['Datetime'][len(df['Datetime']) - 1]
            # print("start_time = {}, end = {}".format(start_time, end))
            # sleep(1.0)
            params.append((col, start_time, end))
            time_start = end
            if time_start >= time_end:
                break
            
    # 使用多线程画图
    # 遍历每一个指标的线程池
    # 遍历每一个指标的画图参数
    # 画图
    for i in range(len(pool)):
        for j in range(len(params)):
            # 参数需要传入col,start,end,df
            pool[i].apply_async(plot, args=(params[j][0], params[j][1], params[j][2], df))
            # 输出调用的函数和参数,忽略df
            # 输出max_thread
            print("max_thread = {}, plot({},{},{}".format(max_thread, params[j][0], params[j][1], params[j][2]))
    # 关闭线程池
    # 等待线程池中的所有线程结束
    for i in range(len(pool)):
        pool[i].close()
        pool[i].join()


# 定义函数，传入指标名称和起始终止时间
# 画出指定指标的图
# 设置图例和标题
# 旋转X轴的时间
# 设置X,Y轴的字体大小为5
# 保存之前将时间格式化为%Y-%m-%d-%H-%M-%S，否则文件名中的:非法
# 输出进度百分比，保留两位小数，每画完一个图就输出一次
def plot(col, start, end_time, df_part):
    global current
    # 画图
    # 从df_part中选出指定时间段的数据
    data = df_part[(df_part['Datetime'] >= start) & (df_part['Datetime'] <= end_time)]
    # 画图
    plt.plot(data['Datetime'], data[col])
    # 设置图例和标题
    plt.legend([col])
    # 标题设置为起始时间和终止时间
    plt.title(start.strftime("%Y-%m-%d %H:%M:%S") + " - " + end_time.strftime("%Y-%m-%d %H:%M:%S"))
    # 旋转X轴的时间
    plt.xticks(rotation=45)
    # 设置X,Y轴的字体大小为5
    plt.tick_params(labelsize=5)
    # 设置图片大小，拉长图片，使得X轴的时间更加清晰
    plt.gcf().set_size_inches(40, 5)
    # 保存之前将时间格式化为%Y-%m-%d-%H-%M-%S，否则文件名中的:非法
    # 将图片分月份保存
    # 文件目录名: ../data/output/original/指标名称/年份/月份
    # 文件名: 指标名称_起始时间_终止时间.png
    dir_name = "../data/output/original/" + col + "/" + start.strftime("%Y") + "/" + start.strftime("%m")
    # 如果目录不存在，就创建目录
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # 保存图片
    # 输出文件名
    # 输出进度百分比，保留两位小数，每画完一个图就输出一次
    filename = dir_name + "/" + col + "_" + start.strftime("%Y-%m-%d-%H-%M-%S") + "_" + end_time.strftime(
        "%Y-%m-%d-%H-%M-%S") + ".png"
    plt.savefig(filename)
    plt.close()
    # 多线程输出当前处理的文件名
    # print(filename)


if __name__ == '__main__':
    plot_all()
