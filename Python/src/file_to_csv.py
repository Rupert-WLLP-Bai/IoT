# Author: Norfloxaciner B
# Description: This file contains a Python script that processes a .txt file and replaces all instances of ";" with ",".
# Version: 1.0
# Date: 2022/12/14

import pandas as pd


def file_to_csv(src_file_path: str, csv_file_path: str):
    """ 处理txt文件并保存为csv文件

    :param src_file_path: txt文件路径
    :param csv_file_path: csv文件路径
    """
    # Open the file and read its contents into a string
    with open(src_file_path, "r") as file:
        original_string = file.read()

    # Replace all instances of ";" with "," in the string
    replaced_string = original_string.replace(";", ",")

    # Open the file and write the contents of the replaced_string to it
    with open(csv_file_path, "w") as file:
        file.write(replaced_string)

    # Read the CSV file using the read_csv() function
    df = pd.read_csv(csv_file_path, dtype={
        'Date': 'str',
        'Time': 'str',
        'Global_active_power': 'str',
        'Global_reactive_power': 'str',
        'Voltage': 'str',
        'Global_intensity': 'str',
        'Sub_metering_1': 'str',
        'Sub_metering_2': 'str',
        'Sub_metering_3': 'str'
    })

    # Check if the DataFrame is empty
    if df.empty:
        print('The CSV file is not valid or there is a problem with the file')
    else:
        print('The CSV file is valid')


if __name__ == "__main__":
    file_to_csv("../data/household_power_consumption.txt", "../data/data.csv")
