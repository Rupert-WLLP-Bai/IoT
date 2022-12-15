# Author: Norfloxaciner B
# Description: This script is used to preprocess the data with missing values
# Version: 1.0
# Date: 2022/12/15

import pandas as pd

# Read the CSV file using the read_csv() function
df = pd.read_csv("../data/data_datetime_with_nan.csv", dtype={
    'Datetime': 'str',
    'Global_active_power': 'str',
    'Global_reactive_power': 'str',
    'Voltage': 'str',
    'Global_intensity': 'str',
    'Sub_metering_1': 'str',
    'Sub_metering_2': 'str',
    'Sub_metering_3': 'str'
})

# Set the 'Datetime' column type to datetime
print("setting Datetime column type to datetime...")
df['Datetime'] = pd.to_datetime(df['Datetime'])
print("setting Datetime column type to datetime done.")

# Replace the missing values '?' with NaN
print("replacing missing values with NaN...")
df.replace('?', pd.NA, inplace=True)
print("replacing missing values with NaN done.")

# Set the rest of the columns type to float
print("setting rest of the columns type to float...")
# use pd.to_numeric() to avoid nan value error
for column in df.columns:
    if column != 'Datetime':
        df[column] = pd.to_numeric(df[column])
print("setting rest of the columns type to float done.")

# Replace NaN with the mean value of the month
# Columns are as follows:
# Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3
# Output the day when NaN is found and the mean value of the month
# Output every one month
# Store the mean value of the month before replacing NaN
# Output how many NaN are replaced
# Use loc to change the original data
# Output the column name and the month which is under checking at every step
print("replacing NaN with the mean value of the month...")
month = 0
month_mean = {}
for column in df.columns:
    if column != 'Datetime':
        print("column: " + column)
        for index, row in df.iterrows():
            if pd.isna(row[column]):
                if row['Datetime'].month != month:
                    month = row['Datetime'].month
                    # print("month: " + str(month))
                    month_mean[column] = df.loc[(df['Datetime'].dt.month == month) & (df[column].notna()), column].mean()
                    # print("mean value of the month: " + str(month_mean[column]))
                df.loc[index, column] = month_mean[column]
            # print("[Column = {}], NaN is found at time: {}".format(column, row['Datetime']))
    print("column: {} done.".format(column))
print("replacing NaN with the mean value of the month done.")

# save the data to a new file
print("saving data to a new file...")
df.to_csv("../data/data_datetime.csv", index=False)
print("saving data to a new file done.")

