# # Author: Norfloxaciner B
# Description: 画图
# Version: 1.0
# Date: 2022/12/14

import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# 根据时间画出每1小时的用电量
# 数据格式:
# Datetime,Global_active_power,Global_reactive_power,Voltage,Global_intensity,Sub_metering_1,Sub_metering_2,Sub_metering_3
# 时间,全局有功功率,全局无功功率,电压,全局电流强度,子电表1,子电表2,子电表3
# 数据起始时间:2006/12/16 17:24
# 数据结束时间:2008/12/13 21:38
# 数据间隔:1分钟
# 每小时画出一个图
# 输出文件路径: ../data/output/{指标名称}
# 输出文件名: {起始时间}-{结束时间}.png
# 设置画图风格
# 设置图例和标题

def plot_all():
    # 读取数据
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
    # 将第一列的数据转换为时间格式
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    # 将后七列的数据转换为浮点数类型
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column])
    # 选择一个指标
    for column in df.columns[1:]:
        # 每小时画出一个图
        for i in range(0, len(df), 60):
            # 画图
            # X轴的时间只输出分钟
            # Y轴的数据为指标的值
            x = df['Datetime'][i:i + 60].dt.strftime("%M")
            y = df[column][i:i + 60]
            sns.lineplot(x=x, y=y)
            # 设置图例和标题
            sns.despine()
            plt.legend([column])
            plt.title(
                f"{df['Datetime'][i].strftime('%Y-%m-%d %H:%M')} ~ {df['Datetime'][i + 59].strftime('%Y-%m-%d %H:%M')}")
            # X轴字体倾斜45度
            plt.xticks(rotation=45, fontsize=5)
            # XY轴字体设置为最小
            plt.yticks(fontsize=5)
            # 设置图片大小为16, 9

            # 使用os.path.join()拼接路径
            # 获取当前文件的路径
            current_path = os.path.dirname(__file__)
            # 输出文件夹在当前文件夹的上一级的data文件夹的output文件夹
            # 获取上一级文件夹的路径
            parent_path = os.path.dirname(current_path)
            # 输出文件夹路径为{上一级文件夹路径}/data/output/{指标名称}
            output_path = os.path.join(parent_path, "data", "output", column)
            # 如果文件夹不存在则创建
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # 输出文件名为{起始时间}-{结束时间}.png
            # 将时间格式化为%Y-%m-%d %H:%M:%S
            start_time = df['Datetime'][i].strftime("%Y-%m-%d %H-%M-%S")
            end_time = df['Datetime'][i + 59].strftime("%Y-%m-%d %H-%M-%S")
            # 输出文件路径为{输出文件夹路径}/{起始时间}-{结束时间}.png
            output_file = os.path.join(output_path, start_time + " ~ " + end_time + ".png")
            # 输出当前处理的文件路径和进度(百分比)
            # 当前文件: {输出文件路径}, 进度: xx.xx%
            print("当前文件: {}, 进度: {:.2f}%".format(output_file, (i + 60) / len(df) * 100))
            # 保存图片
            plt.savefig(output_file)
            # 清空画布
            plt.clf()


if __name__ == '__main__':
    plot_all()
