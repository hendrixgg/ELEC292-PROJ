# Allocated Import libraries

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt

import pickle

from classifier import preprocess, feature_extract

with open('classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)


def classify_data(input_file):
    """
    Function that classifies data from the CSV file.
    """
    # Read acceleration data from the CSV file.
    df = pd.read_csv(input_file)
    # Use the Time column as the index so that 5 second windows can be created easily.
    df_mean = df.set_index('Time (s)')
    df_mean = df_mean.rolling(window=10).mean()
    df_mean.index = pd.to_datetime(df_mean.index, unit='s')

    labels = []
    for interval in df_mean.rolling(window='5.1s'):
        if interval.index[-1] - interval.index[0] < pd.Timedelta('5s'):
            labels.append(None)
            continue
        features = interval.apply(lambda c: pd.DataFrame([c.max(), c.min(), c.max(
        ) - c.min(), c.mean(), c.median(), c.std(), c.var()]).to_numpy().flatten()).to_numpy().flatten()
        labels.append(classifier.predict([features])[0])
    df['labels'] = labels
    return df


# def generate_plot(data):
#     """
#     A function that generates and displays a plot
#     """
#     plt.figure(figsize=(8, 6))
#     plt.plot(data.index, data['labels'], marker='o', linestyle='-', color='b')
#     plt.title('Classification Results')
#     plt.xlabel('Data point')
#     plt.ylabel('Activity')
#     plt.grid(True)
#     plt.show()


def generate_plot(data):
    _, ax1 = plt.subplots(figsize=(10, 10))
    ax2 = ax1.twinx()
    ax2.set_ylim(-0.1, 1.1)
    # acceleration
    ax1.plot(
        data['Time (s)'], data['Linear Acceleration x (m/s^2)'], c='blue', label='x_acc')
    ax1.plot(
        data['Time (s)'], data['Linear Acceleration y (m/s^2)'], c='orange', label='y_acc')
    ax1.plot(
        data['Time (s)'], data['Linear Acceleration z (m/s^2)'], c='red', label='z_acc')
    # classification
    ax2.plot(data['Time (s)'], data['labels'], c='green',
             label='jumping / (not walking)', linestyle='--')

    ax1.legend()
    ax2.legend()
    ax1.set_title(f'Acceleration Data for interval')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('acceleration (m/s)')
    ax2.set_ylabel('classification 1 = jumping, 0 = walking')
    plt.show()


def load_file():
    """
    Functions that loads CSV file and classifies data.
    """
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    output_data = classify_data(file_path)
    generate_plot(output_data)
    # Ask the user where they want to save the output file.
    file_path = filedialog.asksaveasfilename(
        filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    output_data.to_csv(file_path, index=False)


# df = pd.read_csv(
#     "sensor_data\\csv\\anapayaan\\jumping\\back_pant_pocket Acceleration without g 2024-04-02 14-57-08\\processed.csv")
# generate_plot(df)

# Creates a main application window
root = tk.Tk()
root.title("Activity Classifier")
root.geometry("400x400")
# Creates a button to load the file
load_button = tk.Button(root, text="Load CSV File", command=load_file)
load_button.pack(pady=20)

# Runs the main event loop
root.mainloop()
