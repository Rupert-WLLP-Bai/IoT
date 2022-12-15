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

    # Start time is 2006-12-16 17:24:00
    # End time is 2010-11-26 21:02:00
    # Drop the time column and generate it by pandas function
    # Insert the Datetime column to the first column
    print("dropping Date and Time columns...")
    df.drop(columns=['Date', 'Time'], inplace=True)
    print("dropping Date and Time columns done.")
    start_time = pd.to_datetime("2006-12-16 17:24:00")
    end_time = pd.to_datetime("2010-11-26 21:02:00")
    print("generating Datetime column...")
    df.insert(0, 'Datetime', pd.date_range(start_time, end_time, freq='min'))

    # check if the datetime is in order
    print("checking if the datetime is in order...")
    if df['Datetime'].is_monotonic_increasing:
        print("The datetime is in order.")
    else:
        print("The datetime is not in order.")
    print("checking if the datetime is in order done.")

    # save the data to a new file
    df.to_csv("../data/data_datetime_with_nan.csv", index=False)


if __name__ == '__main__':
    main()
