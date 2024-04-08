import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        # Load dataset
        data = pd.read_csv(file_path)

        plt.figure(figsize=(10, 6))

        # Walking Acceleration vs. Time
        plt.subplot(1, 1, 1)
        plt.plot(data['Time (s)'],
                 data['Linear Acceleration x (m/s^2)'], label='X-axis')
        plt.plot(data['Time (s)'],
                 data['Linear Acceleration y (m/s^2)'], label='Y-axis')
        plt.plot(data['Time (s)'],
                 data['Linear Acceleration z (m/s^2)'], label='Z-axis')
        plt.title('Walking Acceleration vs. Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Additional Creative Visualization Ideas
        # 3D Trajectory Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(data['Linear Acceleration x (m/s^2)'], data['Linear Acceleration y (m/s^2)'], data['Linear Acceleration z (m/s^2)'],
                label='Walking')
        ax.plot(data['Linear Acceleration x (m/s^2)'], data['Linear Acceleration y (m/s^2)'], data['Linear Acceleration z (m/s^2)'],
                label='Jumping')
        ax.set_xlabel('X-axis Acceleration')
        ax.set_ylabel('Y-axis Acceleration')
        ax.set_zlabel('Z-axis Acceleration')  # type: ignore
        ax.set_title('3D Trajectory Plot')
        plt.show()

        # Load metadata (assuming metadata.csv is in the same directory as the uploaded CSV file)
        metadata_path = file_path.replace('.csv', '_metadata.csv')
        metadata = pd.read_csv(metadata_path)

        # Visualization of Meta-Data
        # Histogram of Sampling Rates
        plt.hist(metadata['sampling_rate'], bins=10)
        plt.title('Distribution of Sampling Rates')
        plt.xlabel('Sampling Rate')
        plt.ylabel('Frequency')
        plt.show()

        # Box Plots for Sensor Locations
        plt.boxplot([metadata['sensor_x'], metadata['sensor_y'],
                    metadata['sensor_z']], labels=['X', 'Y', 'Z'])
        plt.title('Sensor Locations')
        plt.xlabel('Axis')
        plt.ylabel('Position')
        plt.show()


# Create the main window
root = tk.Tk()
root.title("CSV File Upload")

# Create a button to upload CSV file
upload_button = tk.Button(root, text="Upload CSV File", command=upload_csv)
upload_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
