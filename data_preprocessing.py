import pandas as pd
import numpy as np

def extract_accelerometer_data(csv_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the accelerometer data columns
    # Modify the column names based on your file's actual headers
    accelerometer_data = df[['accelerometerTimestamp_sinceReboot(s)','accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)']]
    accelerometer_data = accelerometer_data.rename(columns={'accelerometerTimestamp_sinceReboot(s)':'timestamp', 'accelerometerAccelerationX(G)':'x', 'accelerometerAccelerationY(G)':'y', 'accelerometerAccelerationZ(G)':'z'})

    accelerometer_data.to_json(r'C:\Users\Raoul\Documents\GitHub\Boxschlag-Klassifikation\accelerometer_data.json', orient='records', lines=True)

    #returns a dataframe
    return accelerometer_data

def extract_gyroscope_data(csv_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the gyroscope data columns
    gyroscope_data = df[['gyroscopeTimestamp_sinceReboot(s)', 'gyroscopeRotationX(rad/s)', 'gyroscopeRotationY(rad/s)', 'gyroscopeRotationZ(rad/s)']]

    # Save the extracted gyroscope data to a new CSV file
    gyroscope_data.to_json('gyroscope_data.json', orient='index')

    return gyroscope_data


def csv_to_dataset_list(path):
    """Reads a csv file from the given path and converts it to a list of datasets.
    Import mechanism is made for the notation style of the smartPunch project.
    If changes are made here make shure to also change the export function(s)!

    Args:
        path (str in raw notation): Filepath including the filename and extension for csv file to import.
                                                            Make shure to use the raw representation to prevent error messages.
                                                            E.g.: 'r"C:\\Users\\tobia\\Downloads\\accdataset_periodMS2000_sampleUS1000.csv"'

        size_of_a_dataset (str): The number of sample values for each dataset. E.g.: 2001

    Returns:
        list: List object containing the datasets as dataframe objects
    """
    ds_rcsv = extract_accelerometer_data(path)
    #ds_rcsv = pd.read_csv(path)
    #ds_rcsv.drop(['Unnamed: 0'], axis=1, inplace=True)
    ds_rcsv.info()

    # get the size of a dataset
    size_of_a_dataset = ds_rcsv.shape[0]

    # try to split dataframe to dataframe array
    number_of_datasets = int(len(ds_rcsv)/size_of_a_dataset)
    ds_buff = np.split(ds_rcsv, number_of_datasets, axis=0)

    # reset the index to 0...period for all datasets
    # also search for errors in the imported datasets
    idxs = 0
    bugs = 0
    for dset in ds_buff:
        dset.reset_index(drop=True, inplace=True)
        if len(dset) != size_of_a_dataset:
            print("Error in dataset: Period length not valid! Check dataset number: {:d} \n Expected length = {:d} , Found length = {:d} .".format(
                idxs, size_of_a_dataset, len(dset)))
            bugs += 1
        idxs += 1

    if bugs == 0:
        print("Data imported without any errors!\n ##### Information: ####\n Imported datasets: {:d}".format(
            number_of_datasets))
    else:
        print("Errors while data import!\n See error message(s) above for more.")
    return ds_buff

