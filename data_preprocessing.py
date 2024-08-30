import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from sktime.annotation.clasp import find_dominant_window_sizes
import logging
# Set logger-level to "error". Not recommed: Important warnings can be overseen
logging.basicConfig(level=logging.ERROR)

def extract_features_from_axis(data):
    """extracts the needet features for the model training
    :param data: datenframe with the columns acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, quan_x, quan_y, quan_z
    :return: new dataframe with the columns x, y, z, one_value
    """
    acc_data = data[['acc_x', 'acc_y', 'acc_z']].copy()
    gyr_data = data[['gyr_x', 'gyr_y', 'gyr_z']].copy()
    quan_data = data[['quan_x', 'quan_y', 'quan_z']].copy()

    pca = PCA(n_components=1)
    t_acc_data = StandardScaler().fit_transform(acc_data)
    t_gyr_data = StandardScaler().fit_transform(gyr_data)
    t_quan_data = StandardScaler().fit_transform(quan_data)

    r_acc_data = pca.fit_transform(t_acc_data)
    r_gyr_data = pca.fit_transform(t_gyr_data)
    r_quan_data = pca.fit_transform(t_quan_data)

    r_acc_data = r_acc_data.flatten()
    r_gyr_data = r_gyr_data.flatten()
    r_quan_data = r_quan_data.flatten()


    r_data = pd.DataFrame({'r_acc': r_acc_data, 'r_gyr': r_gyr_data, 'r_quan': r_quan_data})

    return pd.concat([data, r_data], axis=1)

def extract_data(csv_file_path,startpunkt=0,  nano=True, endpunkt=0):
    """
    Extrct the data from the csv file

    :param csv_file_path: file path to the csv
    :param startpunkt: cut off point at the beginning of the data
    :param nano: if True the timestamp data gets transforned to nanoseconds
    :param endpunkt: cut of point at the end of the data
    :return: a pandas dataframe with the extracted data
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    if endpunkt != 0:
        df = df[startpunkt:endpunkt].copy()
    else:
        df = df[startpunkt:].copy()

    # Extract the accelerometer data columns
    # Modify the column names based on your file's actual headers
    data = df[
        ['accelerometerTimestamp_sinceReboot(s)', 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)',
         'accelerometerAccelerationZ(G)', 'motionRotationRateX(rad/s)', 'motionRotationRateY(rad/s)', 'motionRotationRateZ(rad/s)',
         'motionQuaternionX(R)' ,'motionQuaternionY(R)' ,'motionQuaternionZ(R)']]

    data = data.rename(
        columns={'accelerometerTimestamp_sinceReboot(s)': 'timestamp', 'accelerometerAccelerationX(G)': 'acc_x',
                 'accelerometerAccelerationY(G)': 'acc_y', 'accelerometerAccelerationZ(G)': 'acc_z', 'motionRotationRateX(rad/s)': 'gyr_x',
                 'motionRotationRateY(rad/s)': 'gyr_y', 'motionRotationRateZ(rad/s)': 'gyr_z'
                 , 'motionQuaternionX(R)' : 'quan_x' , 'motionQuaternionY(R)' : 'quan_y' ,'motionQuaternionZ(R)' : 'quan_z'})
    # drop rows with NaN values
    data = data.dropna()
    data = data.reset_index(drop=True)


    # let timestampt column start with 0 and change it from seconds to nanoseconds
    first_timestamp = data['timestamp'][0]
    data['timestamp'] = data['timestamp'] - first_timestamp

    if nano:
        data['timestamp'] = (data['timestamp'] * 1e9)
    return data

def auto_labeling(data, label:str, title = 'Schlaege'):
    """Automatically labels and segments the punches in the given data and transforms them into a dictionary

    :param data: a pandas dataframe with the columns timestamp, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, quan_x, quan_y, quan_z
    :param label: label for the single segments
    :param title: for the visualisation of the peaks
    :return: Dictionary with the labeled punches
    """
    punches = []

    # PCA
    r_data = reduce_dimensionality(data)

    std = r_data['one_value'].std()
    window_size = find_dominant_window_sizes(r_data['one_value'])

    n = len(r_data)/10
    highest_10 = r_data['one_value'].nlargest(int(n)).sum()
    lowest_10 = r_data['one_value'].nsmallest(int(n)).sum()

    # invert the data if the lowest 10 values are more negative than the highest 10 values are positive
    if lowest_10 < 0 and lowest_10*-1 > highest_10:
        r_data['one_value'] = r_data['one_value'] * -1
        #i_data = invert_dataset(data.copy())
        i_data = data.copy()
    else:
        i_data = data.copy()

    #feature extraction
    i_data = extract_features_from_axis(i_data)

    # df with peaks
    p = scipy.signal.find_peaks(r_data['one_value'], height=std, distance=window_size)
    peaks = data.iloc[p[0]]


    peaks = peaks.reset_index()

    # visual checking of the peaks
    print_peaks(r_data=r_data, peaks=peaks, title=title)

    # endpoint of punches
    start = 0
    end = 0

    first_iteration = True
    index = 0

    for idx, row in peaks.iterrows():
        index = idx

        if first_iteration:
            start = 0
            t = (peaks.loc[1, 'timestamp'] + peaks.loc[0, 'timestamp']) // 2
            t_2 = find_nearest_timestamp(i_data, t)
            end = find_index_by_timestamp(i_data, t_2)
            first_iteration = False
        elif idx == len(peaks) - 1:
            start = end + 1
            end = len(i_data) - 1
        else:
            # start = (peaks.loc[idx - 1, 'index'] + peaks.loc[idx, 'index']) // 2
            start = end + 1
            t = (peaks.loc[idx, 'timestamp'] + peaks.loc[idx + 1, 'timestamp']) // 2
            t_2 = find_nearest_timestamp(i_data, t)
            end = find_index_by_timestamp(i_data, t_2)

        ds = data_to_raw(i_data[start:end], label)
        punches.append(ds)

    # for checking the process
    #print(str(index) + '. Durchlauf: start:' + str(start) + ', ende:' + str(end))

    return punches



def jsonData_to_dataset_in_timedifference_us(data):
    """
    !!! slightly modified version of the original function from the smartPunch project https://github.com/smartpunch !!!

    Creates a list of dataframe objects from a given json object. Converts the timestamp col with absolute timestamps in us.
    The last timestamp is the period time in us since the punch started.

    Keyword arguments:
        data            -- JSON Database representation dumped from mongoDB with timestamps in nanoseconds (type: object)

    Returns:
        list            -- List object containing the datasets as dataframe objects with timestamps in 'since last timestamp' format.
    """

    the_cols = ['timestamp', 'label','acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'quan_x', 'quan_y', 'quan_z', 'r_acc', 'r_gyr', 'r_quan']
    the_data = []

    for value in data:
        the_raws = []
        the_indxs = []
        idx = 0
        # raw_time_us = 0
        for raw in value['raws']:
            # raw_time_us += int(raw['timestamp'])//1000
            t = int(raw['timestamp']) // 1000
            the_raws.append(  [int(t), value['label'], raw['acc_x'], raw['acc_y'], raw['acc_z'],raw['gyr_x'], raw['gyr_y'], raw['gyr_z'], raw['quan_x'],
                             raw['quan_y'], raw['quan_z'],raw['r_acc'], raw['r_gyr'], raw['r_quan'],])
            the_indxs.append(idx)
            idx += 1
        the_data.append(pd.DataFrame(the_raws, the_indxs, the_cols))

    return the_data

def normate_dataset_period(periodLengthInMS, samplingRateUS, dataset, interpolationKind='cubic'):
    """
    !!! slightly modified version of the original function from the smartPunch project https://github.com/smartpunch !!!

    Normates the period of a given dataset list. Datasets in dataframe format! Stretches the periods with interpolation method choosen with the interpolationKind parameter. Default: Cubic. Converting mechanism is made for the notation style of the smartPunch project.

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
    interp_period_us = periodLengthInMS * 1000

    # interpolated dataset array
    ds_interp = []

    # define cols for interp. dataset
    the_cols = ['timestamp', 'label', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'quan_x', 'quan_y', 'quan_z', 'r_acc', 'r_gyr', 'r_quan']

    # ds_interp = pd.DataFrame(dataset_raws_storer,np.arange(0,data_size,1),['x','y','z']);
    # print(ds_interp)

    dataset_numb = 0

    total = len(dataset)

    # create and store the interpolation functions
    for idx, ds_el in enumerate(dataset):

        # print('Index: {}'.format(idx))
        if idx > 0:
            print("progress: {:.2f} %".format(idx * 100 / total))
            if idx % 100 == 0:
                print("progress: {:.2f} %".format(idx * 100 / total))

        # copy original datframe object
        # ds_el = ds_el_orig.copy(deep=True)

        # original period length
        orig_period_length_us = ds_el['timestamp'][len(ds_el['timestamp']) - 1]

        # stores the interpolating functions for each dataset element and axis
        f_x = (interp1d(ds_el['timestamp'].values.copy(), ds_el['acc_x'].values.copy(), kind=interpolationKind,
                        bounds_error=False, fill_value=(ds_el['acc_x'][0].copy(), ds_el['acc_x'][0].copy())))
        f_y = (interp1d(ds_el['timestamp'].values.copy(), ds_el['acc_y'].values.copy(), kind=interpolationKind,
                        bounds_error=False, fill_value=(ds_el['acc_y'][0].copy(), ds_el['acc_y'][0].copy())))
        f_z = (interp1d(ds_el['timestamp'].values.copy(), ds_el['acc_z'].values.copy(), kind=interpolationKind,
                        bounds_error=False, fill_value=(ds_el['acc_z'][0].copy(), ds_el['acc_z'][0].copy())))
        gyr_x = (interp1d(ds_el['timestamp'].values.copy(), ds_el['gyr_x'].values.copy(), kind=interpolationKind,
                        bounds_error=False, fill_value=(ds_el['gyr_x'][0].copy(), ds_el['gyr_x'][0].copy())))
        gyr_y = (interp1d(ds_el['timestamp'].values.copy(), ds_el['gyr_y'].values.copy(), kind=interpolationKind,
                        bounds_error=False, fill_value=(ds_el['gyr_y'][0].copy(), ds_el['gyr_y'][0].copy())))
        gyr_z = (interp1d(ds_el['timestamp'].values.copy(), ds_el['gyr_z'].values.copy(), kind=interpolationKind,
                        bounds_error=False, fill_value=(ds_el['gyr_z'][0].copy(), ds_el['gyr_z'][0].copy())))
        quan_x = (interp1d(ds_el['timestamp'].values.copy(), ds_el['quan_x'].values.copy(), kind=interpolationKind,
                        bounds_error=False, fill_value=(ds_el['quan_x'][0].copy(), ds_el['quan_x'][0].copy())))
        quan_y = (interp1d(ds_el['timestamp'].values.copy(), ds_el['quan_y'].values.copy(), kind=interpolationKind,
                         bounds_error=False, fill_value=(ds_el['quan_y'][0].copy(), ds_el['quan_y'][0].copy())))
        quan_z = (interp1d(ds_el['timestamp'].values.copy(), ds_el['quan_z'].values.copy(), kind=interpolationKind,
                          bounds_error=False, fill_value=(ds_el['quan_z'][0].copy(), ds_el['quan_z'][0].copy())))
        r_acc = (interp1d(ds_el['timestamp'].values.copy(), ds_el['r_acc'].values.copy(), kind=interpolationKind,
                          bounds_error=False, fill_value=(ds_el['r_acc'][0].copy(), ds_el['r_acc'][0].copy())))
        r_gyr = (interp1d(ds_el['timestamp'].values.copy(), ds_el['r_gyr'].values.copy(), kind=interpolationKind,
                          bounds_error=False, fill_value=(ds_el['r_gyr'][0].copy(), ds_el['r_gyr'][0].copy())))
        r_quan = (interp1d(ds_el['timestamp'].values.copy(), ds_el['r_quan'].values.copy(), kind=interpolationKind,
                           bounds_error=False, fill_value=(ds_el['r_quan'][0].copy(), ds_el['r_quan'][0].copy())))

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
                dataset_raws_storer.append([int(
                        # dataset_raws_storer.append([5.3, 5.3, 5.3, int(
                        sample_moment), ds_el['label'][0], float(f_x(sample_moment).copy()), float(f_y(sample_moment).copy()),
                                            float(f_z(sample_moment).copy()),float(gyr_x(sample_moment).copy()), float(gyr_y(sample_moment).copy()),
                                            float(gyr_z(sample_moment).copy()),float(quan_x(sample_moment).copy()), float(quan_y(sample_moment).copy()),
                                            float(quan_z(sample_moment).copy()),float(r_acc(sample_moment).copy()), float(r_gyr(sample_moment).copy()),
                                            float(r_quan(sample_moment).copy())])
                sample_moment += samplingRateUS
                the_indxs.append(idx)
                idx += 1
        else:
            # original interval is to long -> center data values and cut borders
            time_to_cutoff_on_each_side = round(
                ((orig_period_length_us - interp_period_us) / 2), 0)
            # print('time_to_cutoff_on_each_side in us: {}'.format(time_to_cutoff_on_each_side))

            # round((time_to_cutoff_on_each_side/samplingRateUS),0)*samplingRateUS
            pseudo_sample_moment = time_to_cutoff_on_each_side
            # print('pseudo_sample_moment in us: {}'.format(pseudo_sample_moment))

            while sample_moment <= interp_period_us:
                dataset_raws_storer.append([int(
                        # dataset_raws_storer.append([5.3, 5.3, 5.3, int(
                        sample_moment), ds_el['label'][0],
                    float(f_x(pseudo_sample_moment).copy()), float(f_y(pseudo_sample_moment).copy()),
                     float(f_z(pseudo_sample_moment).copy()), float(gyr_x(pseudo_sample_moment).copy()), float(gyr_y(pseudo_sample_moment).copy()),
                     float(gyr_z(pseudo_sample_moment).copy()),float(quan_x(pseudo_sample_moment).copy()), float(quan_y(pseudo_sample_moment).copy()),
                    float(quan_z(pseudo_sample_moment).copy()), float(r_acc(pseudo_sample_moment).copy()), float(r_gyr(pseudo_sample_moment).copy()),
                     float(r_quan(pseudo_sample_moment).copy())])
                pseudo_sample_moment += samplingRateUS
                sample_moment += samplingRateUS
                the_indxs.append(idx)
                idx += 1

        # create a new interpolated dataset entry
        ds_interp.append(pd.DataFrame(
            dataset_raws_storer, the_indxs, the_cols))

    return ds_interp


def plot_ds(ds, save_img=False, img_name='Schlag.png', fig_x = 20, fig_y = 10, acc = False, gyr = False, all = False, quan=False):
    """
    Plots the given dataset
    :param ds: the input dataframe
    :param save_img: if True safes the image in the current project
    :param img_name: name of the image if saved
    :param fig_x: 20 by default
    :param fig_y: 10 by default
    :param acc: if True accererometer data is plotted
    :param gyr: if True gyroscope data is plotted
    :param all: if True all data is plotted
    :param quan: if True quaternion data is plotted
    :return:
    """

    acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, quan_x, quan_y, quan_z = [], [], [], [], [], [], [], [], []
    ds_ = ds.to_dict('records')

    for ds_e in ds_:
        acc_x.append(ds_e['acc_x'])
        acc_y.append(ds_e['acc_y'])
        acc_z.append(ds_e['acc_z'])
        gyr_x.append(ds_e['gyr_x'])
        gyr_y.append(ds_e['gyr_y'])
        gyr_z.append(ds_e['gyr_z'])
        quan_x.append(ds_e['quan_x'])
        quan_y.append(ds_e['quan_y'])
        quan_z.append(ds_e['quan_z'])

    t = np.arange(0, len(acc_x))
    if all:
        fig, ax = plt.subplots(figsize=(fig_x, fig_y))
        ax.plot(t, acc_x, label='Accelerometer_x-Achse(G)')
        ax.plot(t, acc_y, label='Accelerometer_y-Achse(G)')
        ax.plot(t, acc_z, label='Accelerometer_z-Achse(G)')
        ax.plot(t, gyr_x, label='Gyroscope_x-Achse(rad/s)')
        ax.plot(t, gyr_y, label='Gyroscope_y-Achse(rad/s)')
        ax.plot(t, gyr_z, label='Gyroscope_z-Achse(rad/s)')
        ax.plot(t, quan_x, label='Quaternion_x-Achse(R)')
        ax.plot(t, quan_y, label='Quaternion_y-Achse(R)')
        ax.plot(t, quan_z, label='Quaternion_z-Achse(R)')


        plt.legend()




    if acc:
        fig, ax = plt.subplots(figsize=(fig_x,fig_y))
        ax.plot(t, acc_x, label= 'acc_x-Achse')
        ax.plot(t, acc_y, label='acc_y-Achse')
        ax.plot(t, acc_z, label='acc_z-Achse')
        ax.set(xlabel='timestamps', ylabel='acceleration',
               title=img_name)
        plt.legend()

    if quan:
        fig, ax = plt.subplots(figsize=(fig_x, fig_y))
        ax.plot(t, quan_x, label='quan_x')
        ax.plot(t, quan_y, label='quan_y')
        ax.plot(t, quan_z, label='quan_z')
        ax.set(xlabel='timestamps', ylabel='quantation',
               title=img_name)
        plt.legend()

    if gyr:
        fig, ay = plt.subplots(figsize=(fig_x, fig_y))
        ay.plot(t, gyr_x, label='gyr_x-Achse')
        ay.plot(t, gyr_y, label='gyr_y-Achse')
        ay.plot(t, gyr_z, label='gyr_z-Achse')
        ay.set(xlabel='timestamps', ylabel='gyroscope',
               title=img_name)
        plt.legend()


    plt.legend()
    if save_img:
        plt.savefig(img_name)
    plt.show()



def data_to_list(raws, axis='x'):
    data = []
    for e in raws:
        data.append(e[axis])
    return data



def prepareDataset(jsnDataset, periodLengthInMS, sampleRateInUS, exportToCSV=False):
    """
    !!! slightly modified version of the original function from the smartPunch project https://github.com/smartpunch !!!

    Reads a (JSON) Dataset and prepares it to get used by the ML algorithms. Returns a list containing dataframe objects for each punch.

    Keyword arguments:
    jsnDataset                 -- the dictionary with raw data
    periodLengthInMS           -- Target period length for each punch in milliseconds (type: number)
    sampleRateInUS             -- Target sample rate in microseconds to interpolate the datapoints (type: number)
    exportToCSV                -- If True, the created and prepared dataset will be exported (type: boolean, default: False)

    Returns:
        list
    """

    ds_orig = jsonData_to_dataset_in_timedifference_us(
        data=jsnDataset)
    ds_equalPeriod = normate_dataset_period(
        periodLengthInMS, sampleRateInUS, ds_orig)
    ds = change_strings_to_numbs(ds_equalPeriod)
    ds_equalPeriod = []
    ds_orig = []
    if exportToCSV:
        fileName = r"prepared_dataset.csv"
        export_list_of_dataframes_to_csv(ds, fileName)
    return ds

def export_list_of_dataframes_to_csv(list_of_datasets, path):
    """
    !!! slightly modified version of the original function from the smartPunch project https://github.com/smartpunch !!!

    Converts a given list of dataframes and converts it to a csv file to store on the given filesystem path.

    Keyword arguments:
        list_of_datasets        --  List object containing dataframe objects (type: list)
        path                    --  Filepath including the filename and extension for the csv file to export. Make shure to use the raw representation to prevent error messages (type: r"string")

    Returns:
        void
    """
    pd.concat(list_of_datasets).to_csv(path)

def change_strings_to_numbs(dataset):
    """
    !!! slightly modified version of the original function from the smartPunch project https://github.com/smartpunch !!!

    Replaces the string values in the cols: hand, annotator and label with numbers.

    Keyword arguments:
        dataset             -- List of dataframe objects containing the datasets (type: list)

    Returns:
        list                -- List object containing the datasets without any strings.
    """
    datasetList = []

    idx = 0
    for dataSet in dataset:
        datasetList.append(dataSet.copy(deep=True))
        # replace label
        datasetList[idx]['label'] = datasetList[idx]['label'].map(
            {'Gerade': 0, 'Kinnhaken': 1, 'Kopfhaken': 2})

        idx = idx + 1
    return datasetList
def listToDataframe(dataSet):
    """
    !!! slightly modified version of the original function from the smartPunch project https://github.com/smartpunch !!!

    Converts the given list object to an pandas DataFrame object containing all punches in one object, seperated by a individual punch_id column

    Keyword arguments:
    dataSet                -- dataset (type: list)

    Returns:
        pandas.core.frame.DataFrame
    """
    df_list = []
    for idx, e in enumerate(dataSet):
        df = e.assign(punch_id=idx)
        df_list.append(df)
    df_res = pd.concat(df_list)
    df_allInOne = df_res.reset_index(drop=True)
    ds = df_allInOne.rename(
        index=str, columns={"x": "a_x", "y": "a_y", "z": "a_z"})
    return ds


def csv_to_dataset_list(path):
    """
    !!! slightly modified version of the original function from the smartPunch project https://github.com/smartpunch !!!

    Reads a csv file from the given path and converts it to a list of datasets.
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
    #ds_rcsv = extract_data(path)
    ds_rcsv = pd.read_csv(path)
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


def data_to_raw(data, label):
    """
    Creates a dictionary with the given data (one punch) and label
    :param data: input pandas df (intendet for one punch)
    :param label: the label for the punch
    :return: a dictionary containing one punch
    """

    data = data.reset_index()
    first_timestamp = data.loc[0, 'timestamp']
    data['timestamp'] = data['timestamp'] - first_timestamp

    if data.empty:
        print('Data is empty')
        return None
    if len(data) == 0:
        periodNS = data.loc[0, 'timestamp']

    else:
        periodNS = data.loc[len(data) - 1, 'timestamp'] - data.loc[0, 'timestamp']

    # create raws
    raws = []
    for idx, row in data.iterrows():
        raw = {'_id': idx, 'timestamp': row['timestamp'], 'acc_x': row['acc_x'], 'acc_y': row['acc_y'], 'acc_z': row['acc_z']
                , 'gyr_x': row['gyr_x'], 'gyr_y': row['gyr_y'], 'gyr_z': row['gyr_z']
                ,'quan_x': row['quan_x'], 'quan_y': row['quan_y'], 'quan_z': row['quan_z']
                , 'r_acc': row['r_acc'], 'r_gyr': row['r_gyr'], 'r_quan': row['r_quan']}
        raws.append(raw)

    # insert raws into dataset
    one_punch = {'label': label, 'count': len(data), 'periodNS': periodNS, 'raws': raws}

    return one_punch


def invert_column(col):
    """
    Inverts the given column
    """
    return col * (-1)
def invert_dataset(data):
    """
    Inverts the given dataset
    """
    data = data.apply(lambda col: invert_column(col) if col.name != 'timestamp' else col)
    return data

def print_peaks(r_data, peaks, title):
    """
    Visualizes the peaks in the given reduced dataset
    :param r_data: reduced data with the colums "one_value"
    :param peaks: list of peaks (index)
    :param title: title of plot
    :return:
    """
    data = r_data[['one_value']].copy()
    annotation_numbers = list(range(1, len(peaks)+1))
    t = np.arange(0, len(data))
    x_scatter = r_data.iloc[peaks['index']]

    y_list = x_scatter['one_value'].tolist()
    x_list = peaks['index'].tolist()

    fig, ax = plt.subplots(figsize=(25, 10))
    ax.plot(t, data['one_value'], label='one_value')
    ax.scatter(x_list, y_list, 30, "blue", label="spikes")
    for i, txt in enumerate(annotation_numbers):
        ax.annotate(txt, (x_list[i]+0.1, y_list[i]+0.1))
    ax.set(title=title)

    plt.show()

def find_nearest_timestamp(df, number):
    """
    Finds the nearest timestamp from the position of number in the given dataset
    """
    absolute_difference_function = lambda timestamp : abs(timestamp - number)
    nearest_timestamp = df['timestamp'].map(absolute_difference_function).idxmin()

    return nearest_timestamp

def find_index_by_timestamp(df, timestamp):
    """
    Finds the index by the given timestamp
    """
    df = df.reset_index().copy()
    absolute_difference_function = lambda index : abs(index - timestamp)
    index = df['index'].map(absolute_difference_function).idxmin()

    return index

def reduce_dimensionality(data):
    """
    Reduces the dimensionality of the given data
    :param data: pandas dataframe with columns acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, quan_x, quan_y, quan_z
    :return: pandas daframe with one column "one_value"
    """
    r_data = data[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z','quan_x', 'quan_y', 'quan_z']].copy()


    pca = PCA(n_components=1)
    std_data = StandardScaler().fit_transform(r_data)
    gsts_array = pca.fit_transform(std_data)
    finalDf = pd.DataFrame(data = gsts_array, columns = ['one_value'])

    return finalDf


