# Allocated Import libraries

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt


def classify_data(input_file):
    """
    Function that classifies data from the CSV file.
    """
    data = pd.read_csv(input_file)
    labels = ['walking' if i % 2 == 0 else 'jumping' for i in range(len(data))]
    data['labels'] = labels
    return data


def generate_plot(data):
    """
    A function that generates and displays a plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(data.index, data['labels'], marker='o', linestyle='-', color='b')
    plt.title('Classification Results')
    plt.xlabel('Data point')
    plt.ylabel('Activity')
    plt.grid(True)
    plt.show()


def plot_interval(interval, label):
    _, ax = plt.subplots(figsize=(10, 10))

    ax.plot(interval[0], c='blue', label='x_acc')
    ax.plot(interval[1], c='orange', label='y_acc')
    ax.plot(interval[2], c='red', label='z_acc')

    ax.set_title(f'Acceleration Data for interval of class {label}')
    ax.set_xlabel('time steps')
    ax.set_ylabel('acceleration (m/s)')
    plt.show()


def load_file():
    """
    Functions that loads CSV file and classifies data.
    """
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        output_data = classify_data(file_path)
        generate_plot(output_data)


# Creates a main application window
root = tk.Tk()
root.title("Activity Classifier")

# Creates a button to load the file
load_button = tk.Button(root, text="Load CSV File", command=load_file)
load_button.pack(pady=20)

# Runs the main event loop
root.mainloop()
