# This file will be used to organize the sensor data into HDF5 files.
import zipfile
import os
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

_hdf5_file = 'sensor_data.hdf5'


def unzip_sensor_data():
    """
    Unzip data from the sensor_data_raw folder and put it into the sensor_data_raw folder
    """
    # Unzip each person's data.
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
    intervals, labels = [], []
    with h5py.File(_hdf5_file, 'w') as f:
        # Create a group for the organized data.
        dataset_group = f.create_group('dataset')
        Train_group = dataset_group.create_group('Train')
        Test_group = dataset_group.create_group('Test')
        # Create a group for each person.
        for person in os.listdir('./sensor_data/csv'):
            person_group = f.create_group(person)
            # Create a dataset for each action (walking, jumping) and phone position. Then, store the data from csv's in it.
            for action in os.listdir(f'./sensor_data/csv/{person}'):
                for i, dir in enumerate(os.listdir(f'./sensor_data/csv/{person}/{action}')):
                    phone_position = dir.split(' ')[0]
                    df = pd.read_csv(
                        f'./sensor_data/csv/{person}/{action}/{dir}/Raw Data.csv', index_col=0)
                    if not type(df) == pd.DataFrame:
                        break
                    # Store sensor data in person group.
                    person_group.create_dataset(
                        f'{action}_{phone_position}', data=df.reset_index())

                    # Store 5 second windows of sensor data in dataset group.
                    df.index = pd.to_datetime(df.index, unit='s')
                    for interval in df.rolling(window='5s'):
                        if interval.empty:
                            continue
                        intervals.append(interval.reset_index())
                        labels.append(action)

        # Split (90%/10%) train vs. test and shuffle.
        X_train, X_test, Y_train, Y_test = train_test_split(
            intervals, labels, test_size=0.1, random_state=42, shuffle=True)
        # Save dataset into HDF5 file.
        for i, (interval, label) in enumerate(zip(X_train, Y_train)):
            Train_group.create_dataset(f"{i}_{label}", data=interval)
        for i, (interval, label) in enumerate(zip(X_test, Y_test)):
            Test_group.create_dataset(f"{i}_{label}", data=interval)


def load_hdf5_train():
    """
    Load the HDF5 file and return the train data.
    """
    with h5py.File(_hdf5_file, 'r') as f:
        # Dropping the first column which is the Time (s).
        train_data = pd.DataFrame.from_records(
            ((1 if n.split('_')[1] == 'jumping' else 0, pd.DataFrame(d[:, 1:]))
                for n, d in f['dataset/Train'].items()), columns=['label', 'interval'])  # type: ignore
    return train_data


def load_hdf5_test():
    """
    Load the HDF5 file and return the test data.
    """
    with h5py.File(_hdf5_file, 'r') as f:
        # Dropping the first column which is the Time (s).
        test_data = pd.DataFrame.from_records(
            ((1 if n.split('_')[1] == 'jumping' else 0, pd.DataFrame(d[:, 1:]))
                for n, d, in f['dataset/Test'].items()), columns=['label', 'interval'])  # type: ignore
    return test_data


if __name__ == "__main__":
    unzip_sensor_data()
    create_hdf5()
