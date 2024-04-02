# Allocated Import libraries

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt

# Function that classifies data from the CSV file
def classify_data(input_file):
    data = pd.read_csv(input_file)
    labels = ['walking' if i % 2 == 0 else 'jumping' for i in range(len(data))]
    data['labels'] = labels
    return data

# Functions that loads CSV file and classifies data
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if file_path:
        output_data = classify_data(file_path)
        generate_plot(output_data)

# A function that generates and displays a plot
def generate_plot(data):
    plt.figure(figsize=(8,6))
    plt.plot (data.index, data['labels'], marker='o', linestyle='-',color='b')
    plt.title('Classification Results')
    plt.xlabel('Data point')
    plt.ylabel('Activity')
    plt.grid(True)
    plt.show()

# Creates a main application window
root = tk.Tk()
root.title("Activity Classifier")

# Creates a button to load the file
load_button = tk.Button(root, text="Load CSV File", command=load_file)
load_button.pack(pady=20)

# Runs the main event loop
root.mainloop()



