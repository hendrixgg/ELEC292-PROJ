# This file will contain basic functions to visualize the data in the HDF5 files.


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import (Axes3D)

    # Load dataset
    data = pd.read_csv('acceleration_data.csv')

    # Sample Acceleration vs. Time Plots
    walking_data = data[data['activity'] == 'walking']
    jumping_data = data[data['activity'] == 'jumping']

    plt.figure(figsize=(10, 6))

    # Walking Acceleration vs. Time
    plt.subplot(2, 1, 1)
    plt.plot(walking_data['time'], walking_data['acceleration_x'], label='X-axis')
    plt.plot(walking_data['time'], walking_data['acceleration_y'], label='Y-axis')
    plt.plot(walking_data['time'], walking_data['acceleration_z'], label='Z-axis')
    plt.title('Walking Acceleration vs. Time')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()

    # Jumping Acceleration vs. Time
    plt.subplot(2, 1, 2)
    plt.plot(jumping_data['time'], jumping_data['acceleration_x'], label='X-axis')
    plt.plot(jumping_data['time'], jumping_data['acceleration_y'], label='Y-axis')
    plt.plot(jumping_data['time'], jumping_data['acceleration_z'], label='Z-axis')
    plt.title('Jumping Acceleration vs. Time')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Additional Creative Visualization Ideas
    # 3D Trajectory Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(walking_data['acceleration_x'], walking_data['acceleration_y'], walking_data['acceleration_z'],
            label='Walking')
    ax.plot(jumping_data['acceleration_x'], jumping_data['acceleration_y'], jumping_data['acceleration_z'],
            label='Jumping')
    ax.set_xlabel('X-axis Acceleration')
    ax.set_ylabel('Y-axis Acceleration')
    ax.set_zlabel('Z-axis Acceleration')
    ax.set_title('3D Trajectory Plot')
    plt.show()

    # Load metadata
    metadata = pd.read_csv('metadata.csv')

    # Visualization of Meta-Data
    # Histogram of Sampling Rates
    plt.hist(metadata['sampling_rate'], bins=10)
    plt.title('Distribution of Sampling Rates')
    plt.xlabel('Sampling Rate')
    plt.ylabel('Frequency')
    plt.show()

    # Box Plots for Sensor Locations
    plt.boxplot([metadata['sensor_x'], metadata['sensor_y'], metadata['sensor_z']], labels=['X', 'Y', 'Z'])
    plt.title('Sensor Locations')
    plt.xlabel('Axis')
    plt.ylabel('Position')
    plt.show()

