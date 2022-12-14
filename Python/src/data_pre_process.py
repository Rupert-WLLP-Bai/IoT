# Author: Norfloxaciner B
# Description: 数据处理
# Version: 1.0
# Date: 2022/12/14

import pandas as pd


def main():
    # Read the CSV file using the read_csv() function
    df = pd.read_csv("../data/data.csv", dtype={
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
    # Combine 'Date' and 'Time' into 'Datetime'
    print("inserting Datetime column...")
    df.insert(0, 'Datetime', pd.to_datetime(df['Date'] + ' ' + df['Time']))
    print("inserting Datetime column done.")

    print("dropping Date and Time columns...")
    df.drop(columns=['Date', 'Time'], inplace=True)
    print("dropping Date and Time columns done.")

    # save the data to a new file
    df.to_csv("../data/data_datetime_with_nan.csv", index=False)


if __name__ == '__main__':
    main()
