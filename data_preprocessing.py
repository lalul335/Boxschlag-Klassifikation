import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def extract_accelerometer_data(csv_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the accelerometer data columns
    # Modify the column names based on your file's actual headers
    accelerometer_data = df[['accelerometerTimestamp_sinceReboot(s)','accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)']]
    accelerometer_data = accelerometer_data.rename(columns={'accelerometerTimestamp_sinceReboot(s)':'timestamp', 'accelerometerAccelerationX(G)':'x', 'accelerometerAccelerationY(G)':'y', 'accelerometerAccelerationZ(G)':'z'})
    accelerometer_data['label'] = 0
    # drop rows with NaN values
    accelerometer_data = accelerometer_data.dropna()
    accelerometer_data = accelerometer_data.reset_index(drop=True)

    # let timestampt column start with 0 and change it from seconds to nanoseconds
    first_timestamp = accelerometer_data['timestamp'][0]
    accelerometer_data['timestamp'] = accelerometer_data['timestamp'] - first_timestamp
    accelerometer_data['timestamp'] = accelerometer_data['timestamp'] * 1e9

    #accelerometer_data.to_json(r'C:\Users\Raoul\Documents\GitHub\Boxschlag-Klassifikation\accelerometer_data.json', orient='records', lines=True)

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


def csv_to_json(file_path, ds=[]):
    # Read the CSV file and morph it into a indexed Dictionary
    data = extract_accelerometer_data(file_path)


    # create raws
    raws = []
    for idx, row in data.iterrows():
        raw = {'_id': idx, 'timestamp': row['timestamp'], 'x': row['x'], 'y': row['y'], 'z': row['z']}
        raws.append(raw)

    # insert raws into dataset
    dataset = {'raws': raws, 'label': 0, 'count': len(data)}

    # append dataset to dataset list
    ds.append(dataset)

    return ds


def normate_dataset_period(periodLengthInMS, samplingRateUS, ds_el, interpolationKind='cubic'):
    """Normates the period of a given dataset list. Datasets in dataframe format! Stretches the periods with interpolation method choosen with the interpolationKind parameter. Default: Cubic. Converting mechanism is made for the notation style of the smartPunch project.

    Keyword arguments:
        periodLengthInMS            --  Required period length in [ms] for each dataset (type: number)
        samplingRateUS              --  Sampling rate in [us] for each datapoint (type: number)
        dataset                     --  List of dataframe objects containing the datasets (type: list)
        interpolationKind           --  Interpolation method (type: string, default: 'cubic', available methods: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
                                        'previous', 'next', where 'zero', 'slinear', 'quadratic' or 'cubic')

    Returns:
        list        -- List object containing the interpolated datasets as dataframe objects with the stretched/compressed time period.
    """

    print("Starting new normalization/interpolation...")

    # interpolated period length in us
    interp_period_us = periodLengthInMS*1000

    # interpolated dataset array
    ds_interp = []

    # define cols for interp. dataset
    the_cols = ['timestamp', 'x', 'y', 'z','label']

    # ds_interp = pd.DataFrame(dataset_raws_storer,np.arange(0,data_size,1),['x','y','z']);
    # print(ds_interp)

    dataset_numb = 0

    total = len(dataset)



    # copy original datframe object
    # ds_el = ds_el_orig.copy(deep=True)

    # original period length
    orig_period_length_us = ds_el['timestamp'][len(ds_el['timestamp'])-1]

    # stores the interpolating functions for each dataset element and axis
    f_x = (interp1d(ds_el['timestamp'].values.copy(), ds_el['x'].values.copy(), kind=interpolationKind,
                        bounds_error=False, fill_value=(ds_el['x'][0].copy(), ds_el['x'][0].copy())))
    f_y = (interp1d(ds_el['timestamp'].values.copy(), ds_el['y'].values.copy(), kind=interpolationKind,
                        bounds_error=False, fill_value=(ds_el['y'][0].copy(), ds_el['y'][0].copy())))
    f_z = (interp1d(ds_el['timestamp'].values.copy(), ds_el['z'].values.copy(), kind=interpolationKind,
                        bounds_error=False, fill_value=(ds_el['z'][0].copy(), ds_el['z'][0].copy())))

    # helper variables
    sample_moment = 0

    # stores the interpolated raw data of the current dataset element as array(s)
    dataset_raws_storer = []
    the_indxs = []
    idx = 0

    # original period length is in intervall limit
    if orig_period_length_us <= interp_period_us:
          # interpolate x,y,z raw array of all elements in current dataset
         while sample_moment <= interp_period_us:
               dataset_raws_storer.append([float(f_x(sample_moment).copy()), float(f_y(sample_moment).copy()), float(f_z(sample_moment).copy()), int(
                   # dataset_raws_storer.append([5.3, 5.3, 5.3, int(
                   sample_moment), ds_el['label'][0], ds_el['hand'][0], ds_el['annotator'][0]])
               sample_moment += samplingRateUS
               the_indxs.append(idx)
               idx += 1
    else:
        # original interval is to long -> center data values and cut borders
        time_to_cutoff_on_each_side = round(
            ((orig_period_length_us - interp_period_us)/2), 0)
        # print('time_to_cutoff_on_each_side in us: {}'.format(time_to_cutoff_on_each_side))
        # round((time_to_cutoff_on_each_side/samplingRateUS),0)*samplingRateUS
        pseudo_sample_moment = time_to_cutoff_on_each_side
        # print('pseudo_sample_moment in us: {}'.format(pseudo_sample_moment))
        while sample_moment <= interp_period_us:
            dataset_raws_storer.append([float(f_x(pseudo_sample_moment).copy()), float(f_y(pseudo_sample_moment).copy()), float(f_z(pseudo_sample_moment).copy()),
                                        # dataset_raws_storer.append([5.3, 5.3, 5.3,
                                        int(sample_moment), ds_el['label'][0], ds_el['hand'][0], ds_el['annotator'][0]])
            pseudo_sample_moment += samplingRateUS
            sample_moment += samplingRateUS
            the_indxs.append(idx)
            idx += 1
        #create a new interpolated dataset entry
    ds_interp.append(pd.DataFrame(
            dataset_raws_storer, the_indxs, the_cols))

    return ds_interp

def jsonData_to_dataset_in_timedifference_us(data):
    """Creates a list of dataframe objects from a given json object. Converts the timestamp col with absolute timestamps in us.
       The last timestamp is the period time in us since the punch started.

    Keyword arguments:
        data            -- JSON Database representation dumped from mongoDB with timestamps in nanoseconds (type: object)

    Returns:
        list            -- List object containing the datasets as dataframe objects with timestamps in 'since last timestamp' format.
    """

    the_cols = ['timestamp', 'x', 'y', 'z',  'label']
    the_data = []



    for value in data:
        the_raws = []
        the_indxs = []
        idx = 0
        raw_time_us = 0
        for raw in value['raws']:
            raw_time_us += int(raw['timestamp'])/1000
            the_raws.append([raw['x'], raw['y'], raw['z'], int(
                raw_time_us), value['label'], value['hand'], value['annotator']])
            the_indxs.append(idx)
            idx += 1
        the_data.append(pd.DataFrame(the_raws, the_indxs, the_cols))
    return the_data