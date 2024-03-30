# This file will be used to organize the sensor data into HDF5 files.
import zipfile
import os
import h5py
import pandas as pd


def unzip_sensor_data():
    """
    Unzip data from the sensor_data_raw folder and put it into the sensor_data_raw folder
    """
    # Unzip each person's data
    for dir in os.listdir('./sensor_data_raw'):
        # Unzip data for each action.
        for dir_type in ['walking', 'jumping']:
            current_dir = './sensor_data_raw/' + dir + '/' + dir_type
            # Unzip each file in the directory.
            for file in os.listdir(current_dir):
                if file.endswith('.zip'):
                    with zipfile.ZipFile(current_dir + '/' + file) as zip_ref:
                        # Extract all the contents (including metadata)
                        zip_ref.extractall(
                            './sensor_data/csv/' + dir + '/' + dir_type + '/' + file[:-4])


def create_hdf5():
    """
    Create one HDF5 file for all the data.
    """
    with h5py.File('sensor_data.hdf5', 'w') as f:
        # Create a group for each person.
        for person in os.listdir('./sensor_data/csv'):
            person_group = f.create_group(person)
            # Create a dataset for each action (walking, jumping) and phone position. Then, store the data from csv's in it.
            for action in os.listdir('./sensor_data/csv/' + person):
                for dir in os.listdir('./sensor_data/csv/' + person + '/' + action):
                    phone_position = dir.split(' ')[0]
                    df = pd.read_csv(
                        './sensor_data/csv/' + person + '/' + action + '/' + dir)
                    person_group.create_dataset(
                        f'{action}_{phone_position}', data=df)


if __name__ == "__main__":
    unzip_sensor_data()
    create_hdf5()
