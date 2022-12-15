# Author: Norfloxaciner B
# Description: 预测所有的指标，将结果存放在csv文件中
# Version: 1.0
# Date: 2022/12/15

import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool, cpu_count, Manager, Process
import seaborn as sns
from matplotlib import pyplot as plt


# Read CSV file
def read_data():
    """
        :return: dataframe
    """
    path = "../data/data_datetime.csv"
    dtype = {
        'Datetime': 'str',
        'Global_active_power': 'float',
        'Global_reactive_power': 'float',
        'Voltage': 'float',
        'Global_intensity': 'float',
        'Sub_metering_1': 'float',
        'Sub_metering_2': 'float',
        'Sub_metering_3': 'float'
    }
    data = pd.read_csv(path, dtype=dtype)
    print("len(df) = {}".format(len(data)))
    return data.head(1440).copy()


# Build model
def build_model():
    """
        :return: model
    """
    # use LSTM to build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=[None, 1]),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)
    ])
    # compile model
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9),
                  metrics=["mae"])
    return model


# Split data into train set and test set
# train_size = 0.8
# test_size = 0.2
# args: dataframe, train_size, test_size, column_name
def split_data(dataframe, column_name, train_size=0.8, test_size=0.2):
    """
        :param dataframe: dataframe
        :param train_size: float
        :param test_size: float
        :param column_name: str
        :return: train_set, test_set
    """
    train_size = int(len(dataframe) * 0.8)
    test_size = len(dataframe) - train_size
    train, test = dataframe.iloc[0:train_size], dataframe.iloc[train_size:len(dataframe)]
    train_set = train[column_name].values
    test_set = test[column_name].values
    # convert to tensorflow dataset
    train_set = tf.data.Dataset.from_tensor_slices(train_set)
    test_set = tf.data.Dataset.from_tensor_slices(test_set)
    # set window size
    # set batch size
    # set prefetch
    # set predict step
    window_size = 60
    batch_size = 100
    predict_step = 1
    train_set = train_set.window(window_size + predict_step, shift=1, drop_remainder=True)
    train_set = train_set.flat_map(lambda window: window.batch(window_size + predict_step))
    train_set = train_set.shuffle(1000).map(lambda window: (window[:-predict_step], window[-predict_step:]))
    train_set = train_set.batch(batch_size).prefetch(1)
    test_set = test_set.window(window_size + predict_step, shift=1, drop_remainder=True)
    test_set = test_set.flat_map(lambda window: window.batch(window_size + predict_step))
    test_set = test_set.map(lambda window: (window[:-predict_step], window[-predict_step:]))
    test_set = test_set.batch(batch_size).prefetch(1)
    return train_set, test_set


# Train model
def train_model(model, train_set, test_set, column_name):
    """
        :param model: model
        :param train_set: tensorflow dataset
        :param test_set: tensorflow dataset
        :param column_name: str
        :return: model
    """
    # set checkpoint
    checkpoint_path = "../data/output/checkpoint/{}.ckpt".format(column_name)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True)
    # train model
    history = model.fit(train_set, epochs=10, validation_data=test_set, callbacks=[checkpoint], verbose=1)
    return model


# Predict
def predict(model, test_set, column_name):
    """
        :param model: model
        :param test_set: tensorflow dataset
        :param column_name: str
        :return: predict
    """
    # load checkpoint
    checkpoint_path = "../data/output/checkpoint/{}.ckpt".format(column_name)
    model.load_weights(checkpoint_path)
    # predict
    predict = model.predict(test_set)
    return predict

# save predict result to excel file 
def save_predict(predict, column_name):
    """
        :param predict: numpy array
        :param column_name: str
        :return: None
    """
    # convert to numpy array
    predict = predict.reshape(-1)
    # save to excel file
    path = "../data/output/predict/{}.xlsx".format(column_name)
    # save the actual value and predict value to two columns
    df = pd.DataFrame({"actual": predict})
    df.to_excel(path, index=False)
    
    

# Plot
def plot(predict, test_set, column_name):
    """
        :param predict: numpy array
        :param test_set: tensorflow dataset
        :param column_name: str
        :return: None
    """
    # convert to numpy array
    predict = predict.reshape(-1)
    test_set = test_set.unbatch()
    test_set = test_set.batch(60).prefetch(1)
    test_set = list(test_set.as_numpy_iterator())
    test_set = np.array(test_set)
    test_set = test_set.reshape(-1)
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(test_set, label="test_set")
    plt.plot(predict, label="predict")
    plt.legend(loc="upper left")
    plt.title(column_name)
    plt.savefig("../data/output/result/{}.png".format(column_name))
    plt.show()


# Main
if __name__ == "__main__":
    # read data
    data = read_data()
    # build model
    model = build_model()
    # split data
    train_set, test_set = split_data(data, "Global_active_power")
    # Y labels
    y_labels = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1",
                "Sub_metering_2", "Sub_metering_3"]
    # train model
    for y_label in y_labels:
        train_set, test_set = split_data(data, y_label)
        model = train_model(model, train_set, test_set, y_label)
        predict = predict(model, test_set, y_label)
        save_predict(predict, y_label)
        # plot(predict, test_set, y_label)
