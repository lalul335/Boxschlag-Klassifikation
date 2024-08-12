from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import random
import matplotlib.cm as cm
import pickle

# tsfresh modules (for feature extraction)
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
import logging
# Set logger-level to "error". Not recommed: Important warnings can be overseen
logging.basicConfig(level=logging.ERROR)


def extract_data(csv_file_path,startpunkt=0,  nano=True, endpunkt=0, rechts=False):
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
         'motionYaw(rad)','motionRoll(rad)','motionPitch(rad)']]

    data = data.rename(
        columns={'accelerometerTimestamp_sinceReboot(s)': 'timestamp', 'accelerometerAccelerationX(G)': 'acc_x',
                 'accelerometerAccelerationY(G)': 'acc_y', 'accelerometerAccelerationZ(G)': 'acc_z', 'motionRotationRateX(rad/s)': 'gyr_x',
                 'motionRotationRateY(rad/s)': 'gyr_y', 'motionRotationRateZ(rad/s)': 'gyr_z', 'motionYaw(rad)': 'yaw', 'motionRoll(rad)': 'roll', 'motionPitch(rad)': 'pitch'})
    # drop rows with NaN values
    data = data.dropna()
    data = data.reset_index(drop=True)

    if rechts:
        data = invert_dataset(data)

    # let timestampt column start with 0 and change it from seconds to nanoseconds
    first_timestamp = data['timestamp'][0]
    data['timestamp'] = data['timestamp'] - first_timestamp
    if nano:
        data['timestamp'] = (data['timestamp'] * 1e9)
    return data

def invert_column(col):
    return col * (-1)
def invert_dataset(data):
    data = data.apply(lambda col: invert_column(col) if col.name != 'timestamp' else col)
    return data

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
    ds_rcsv = extract_data(path)
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

def data_to_json(data, label):
    # create raws
    raw = []
    for idx, row in data.iterrows():
        raw = {'_id': idx, 'timestamp': row['timestamp'], 'x': row['x'], 'y': row['y'], 'z': row['z']}
        raw.append(raw)

    # insert raws into dataset
    one_punch = {'label': label, 'count': len(data), 'periodNS': data['timestamp'].sum() ,'raws': raw}

    return one_punch

def data_to_raw(data, label):
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
               , 'gyr_x': row['gyr_x'], 'gyr_y': row['gyr_y'], 'gyr_z': row['gyr_z'], 'yaw': row['yaw'], 'roll': row['roll'], 'pitch': row['pitch']}
        raws.append(raw)

    # insert raws into dataset
    one_punch = {'label': label, 'count': len(data), 'periodNS': periodNS, 'raws': raws}

    return one_punch

def auto_labeling(data, height, distance, label, invert=False):
    # liste mit Schlägen
    punches = []

    # PCA
    r_data = reduce_dimensionality(data)

    # df mit den Hochpunkten
    p = scipy.signal.find_peaks(r_data['one_value'], height=height, distance=distance)
    peaks = data.iloc[p[0]]

    # eine Spalte mit den Indizes der Hochpunkte wird erstellt
    peaks = peaks.reset_index()

    # invert after finding peaks
    if invert:
        i_data = invert_dataset(data.copy())
    else:
        i_data = data.copy()

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


    print(str(index) + '. Durchlauf: start:' + str(start) + ', ende:' + str(end))

    return punches

def find_nearest_timestamp(df, number):
    absolute_difference_function = lambda timestamp : abs(timestamp - number)
    nearest_timestamp = df['timestamp'].map(absolute_difference_function).idxmin()
    #return df.loc[nearest_timestamp, ['timestamp']]
    return nearest_timestamp

def find_index_by_timestamp(df, timestamp):
    df = df.reset_index().copy()
    index = df.loc[df['timestamp'] == timestamp].index
    absolute_difference_function = lambda index : abs(index - timestamp)
    index = df['index'].map(absolute_difference_function).idxmin()

    return index

def reduce_dimensionality(data):
    r_data = data[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'yaw', 'roll', 'pitch']].copy()

    pca = PCA(n_components=1)
    std_data = StandardScaler().fit_transform(r_data)
    gsts_array = pca.fit_transform(std_data)
    finalDf = pd.DataFrame(data = gsts_array, columns = ['one_value'])

    return finalDf

def jsonData_to_dataset_in_timedifference_us(data):
    """Creates a list of dataframe objects from a given json object. Converts the timestamp col with absolute timestamps in us.
       The last timestamp is the period time in us since the punch started.

    Keyword arguments:
        data            -- JSON Database representation dumped from mongoDB with timestamps in nanoseconds (type: object)

    Returns:
        list            -- List object containing the datasets as dataframe objects with timestamps in 'since last timestamp' format.
    """

    the_cols = ['x', 'y', 'z', 'timestamp', 'label']
    the_data = []

    for value in data:
        the_raws = []
        the_indxs = []
        idx = 0
        # raw_time_us = 0
        for raw in value['raws']:
            # raw_time_us += int(raw['timestamp'])//1000
            t = int(raw['timestamp']) // 1000
            the_raws.append([raw['x'], raw['y'], raw['z'], int(
                t), value['label']])
            the_indxs.append(idx)
            idx += 1
        the_data.append(pd.DataFrame(the_raws, the_indxs, the_cols))

    return the_data

def normate_dataset_period(periodLengthInMS, samplingRateUS, dataset, interpolationKind='cubic'):
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
    interp_period_us = periodLengthInMS * 1000

    # interpolated dataset array
    ds_interp = []

    # define cols for interp. dataset
    the_cols = ['x', 'y', 'z', 'timestamp', 'label']

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
                dataset_raws_storer.append([float(f_x(sample_moment).copy()), float(f_y(sample_moment).copy()),
                                            float(f_z(sample_moment).copy()), int(
                        # dataset_raws_storer.append([5.3, 5.3, 5.3, int(
                        sample_moment), ds_el['label'][0]])
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
                dataset_raws_storer.append(
                    [float(f_x(pseudo_sample_moment).copy()), float(f_y(pseudo_sample_moment).copy()),
                     float(f_z(pseudo_sample_moment).copy()),
                     # dataset_raws_storer.append([5.3, 5.3, 5.3,
                     int(sample_moment), ds_el['label'][0]])
                pseudo_sample_moment += samplingRateUS
                sample_moment += samplingRateUS
                the_indxs.append(idx)
                idx += 1

        # create a new interpolated dataset entry
        ds_interp.append(pd.DataFrame(
            dataset_raws_storer, the_indxs, the_cols))
    print('Fertig, jetzt erstmal lecker Bierchen!')
    return ds_interp


def plot_ds(ds, save_img=False, img_name='Schlag.png', fig_x = 20, fig_y = 10, acc = True, gyr = True, rot = True, all = False):
    acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, yaw, roll ,pitch = [], [], [], [], [], [], [], [], []
    ds_ = ds.to_dict('records')

    for ds_e in ds_:
        acc_x.append(ds_e['acc_x'])
        acc_y.append(ds_e['acc_y'])
        acc_z.append(ds_e['acc_z'])
        gyr_x.append(ds_e['gyr_x'])
        gyr_y.append(ds_e['gyr_y'])
        gyr_z.append(ds_e['gyr_z'])
        yaw.append(ds_e['yaw'])
        roll.append(ds_e['roll'])
        pitch.append(ds_e['pitch'])

    t = np.arange(0, len(acc_x))
    if all:
        fig, ax = plt.subplots(figsize=(fig_x, fig_y))
        ax.plot(t, acc_x, label='acc_x-Achse')
        ax.plot(t, acc_y, label='acc_y-Achse')
        ax.plot(t, acc_z, label='acc_z-Achse')
        ax.plot(t, gyr_x, label='gyr_x-Achse')
        ax.plot(t, gyr_y, label='gyr_y-Achse')
        ax.plot(t, gyr_z, label='gyr_z-Achse')
        ax.plot(t, yaw, label='yaw')
        ax.plot(t, roll, label='roll')
        ax.plot(t, pitch, label='pitch')
        ax.set(xlabel='timestamps', ylabel='acceleration',
               title=img_name)
        plt.legend()
    if acc:
        fig, ax = plt.subplots(figsize=(fig_x,fig_y))
        ax.plot(t, acc_x, label= 'acc_x-Achse')
        ax.plot(t, acc_y, label='acc_y-Achse')
        ax.plot(t, acc_z, label='acc_z-Achse')
        ax.set(xlabel='timestamps', ylabel='acceleration',
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

    if rot:
        fig, az = plt.subplots(figsize=(fig_x, fig_y))
        az.plot(t, yaw, label='yaw')
        az.plot(t, roll, label='roll')
        az.plot(t, pitch, label='pitch')
        az.set(xlabel='timestamps', ylabel='rotation',
               title=img_name)





    plt.legend()
    if save_img:
        plt.savefig(img_name)
    plt.show()


def plot_ds_element(ds_e, save_img=False, img_name='Ein_Schlag.png',):
    raws = ds_e['raws']
    raws[0]
    acc_x = data_to_list2(raws, axis='acc_x')
    acc_y = data_to_list2(raws, axis='acc_y')
    acc_z = data_to_list2(raws, axis='acc_z')

    t = np.arange(0, len(acc_x))

    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(t, acc_x, label='acc_x')
    ax.plot(t, acc_y, label='acc_y')
    ax.plot(t, acc_z, label='acc_z')

    ax.set(xlabel='datastamps', ylabel='acceleration',
           title='single dataset element of type: '+ds_e['label'])
    ax.grid()
    plt.legend()
    if save_img:
        plt.savefig(img_name)
    plt.show()

def data_to_list(raws, axis='x'):
    data = []
    for e in raws:
        data.append(e[axis])
    return data

def data_to_list2(raws, axis='x'):
    return [e[axis] for e in raws]

def universal_plotter_for_all_axis(list_of_datasets,
                                   list_of_dataset_legend_titles,
                                   plot_title, y_axis_label, x_axis_label,
                                   figSizeTupel=(20, 10), saveImage=False,
                                   imageName='unknown_image.png'):
    """Plots all axis of multiple datasets (punches) of an given array. Converting mechanism is made for the notation style of the smartPunch project.

    Keyword arguments:
        list_of_datasets                -- List of datasets (type: list)
        list_of_dataset_legend_titles   -- List strings containing the legend titles for each dataset axis (type: list)
        plot_title                      -- Plot title (type: string)
        y_axis_label                    -- Y-Axis label (type: string)
        x_axis_label                    -- X-Axis label (type: string)
        figSizeTupel                    -- Figure size as optional parameter (type: tupel, default: (20,10))
        saveImage                       -- If True the image is saved on the filesystem (type: boolean, default: False)
        imageName                       -- If saveImage is True, the image is saved using the given imageName (type: string, default: "unknown_image.png")

    Returns:
        void
    """

    fig, ax = plt.subplots(figsize=figSizeTupel)

    idx = 0

    for curDataSet in list_of_datasets:
        universal_label = '%(titl)s (x-axis)' % {
            'titl': list_of_dataset_legend_titles[idx]} if list_of_dataset_legend_titles[idx] != '' else 'x-axis'
        ax.plot(curDataSet['timestamp'].values,
                curDataSet['x'], label=universal_label)
        universal_label = '%(titl)s (y-axis)' % {
            'titl': list_of_dataset_legend_titles[idx]} if list_of_dataset_legend_titles[idx] != '' else 'y-axis'
        ax.plot(curDataSet['timestamp'].values,
                curDataSet['y'], label=universal_label)
        universal_label = '%(titl)s (z-axis)' % {
            'titl': list_of_dataset_legend_titles[idx]} if list_of_dataset_legend_titles[idx] != '' else 'z-axis'
        ax.plot(curDataSet['timestamp'].values,
                curDataSet['z'], label=universal_label)
        idx += 1

    ax.set(xlabel=x_axis_label, ylabel=y_axis_label,
           title=plot_title)
    ax.grid()
    plt.legend()
    if saveImage:
        if imageName.endswith('.png'):
            plt.savefig(imageName)
        else:
            if "." not in imageName:
                name = imageName+'.png'
                plt.savefig(name)
            else:
                print("Error: File extension {} not allowed! See docs for more.".format(
                    imageName.split(".")[1]))
    plt.show()

def single_plot_all_axis(dataset, plot_title, y_axis_label, x_axis_label, default_axis_legend=True, figSizeTupel=(20, 10), saveImage=False,
                         imageName='unknown_image.png'):
    """Plots all axis of a single dataset (punch). Converting mechanism is made for the notation style of the smartPunch project.

    Keyword arguments:
        dataset                         -- Dataset (punch) to plot (type: pandas.Dataframe)
        default_axis_legend             -- If not False this parameter is used for the axis legend title instead of the default value (type: string, default: True)
        plot_title                      -- Plot title (type: string)
        y_axis_label                    -- Y-Axis label (type: string)
        x_axis_label                    -- X-Axis label (type: string)
        figSizeTupel                    -- Figure size as optional parameter (type: tupel, default: (20,10))
        saveImage                       -- If True the image is saved on the filesystem (type: boolean, default: False)
        imageName                       -- If saveImage is True, the image is saved using the given imageName (type: string, default: "unknown_image.png")

    Returns:
        void
    """
    fig, ax = plt.subplots(figsize=figSizeTupel)

    universal_label = 'x-axis' if default_axis_legend == True else default_axis_legend[0]
    ax.plot(dataset['timestamp'].values, dataset['x'], label=universal_label)
    universal_label = 'y-axis' if default_axis_legend == True else default_axis_legend[1]
    ax.plot(dataset['timestamp'].values, dataset['y'], label=universal_label)
    universal_label = 'z-axis' if default_axis_legend == True else default_axis_legend[2]
    ax.plot(dataset['timestamp'].values, dataset['z'], label=universal_label)

    ax.set(xlabel=x_axis_label, ylabel=y_axis_label,
           title=plot_title)
    ax.grid()
    plt.legend()
    if saveImage:
        if imageName.endswith('.png'):
            plt.savefig(imageName)
        else:
            if "." not in imageName:
                name = imageName+'.png'
                plt.savefig(name)
            else:
                print("Error: File extension {} not allowed! See docs for more.".format(
                    imageName.split(".")[1]))
    plt.show()

def prepareDataset(datasetPath, periodLengthInMS, sampleRateInUS, exportToCSV=False):
    """Reads a (JSON) Dataset and prepares it to get used by the ML algorithms. Returns a list containing dataframe objects for each punch.

    Keyword arguments:
    datasetPath                -- Path to the dataset (type: string)
    periodLengthInMS           -- Target period length for each punch in milliseconds (type: number)
    sampleRateInUS             -- Target sample rate in microseconds to interpolate the datapoints (type: number)
    exportToCSV                -- If True, the created and prepared dataset will be exported (type: boolean, default: False)

    Returns:
        list
    """
    with open(datasetPath, 'r') as f:
        jsnDataset = json.load(f)
    ds_orig = jsonData_to_dataset_in_timedifference_us(
        data=jsnDataset)
    ds_equalPeriod = normate_dataset_period(
        periodLengthInMS, sampleRateInUS, ds_orig)
    ds = change_strings_to_numbs(ds_equalPeriod)
    ds_equalPeriod = []
    ds_orig = []
    if exportToCSV:
        fileName = r"dataset_periodMS" + \
            str(periodLengthInMS)+"_sampleUS"+str(sampleRateInUS)+".csv"
        export_list_of_dataframes_to_csv(ds, fileName)
    return ds

def export_list_of_dataframes_to_csv(list_of_datasets, path):
    """Converts a given list of dataframes and converts it to a csv file to store on the given filesystem path.

    Keyword arguments:
        list_of_datasets        --  List object containing dataframe objects (type: list)
        path                    --  Filepath including the filename and extension for the csv file to export. Make shure to use the raw representation to prevent error messages (type: r"string")

    Returns:
        void
    """
    pd.concat(list_of_datasets).to_csv(path)

def change_strings_to_numbs(dataset):
    """Replaces the string values in the cols: hand, annotator and label with numbers.

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
    """Converts the given list object to an pandas DataFrame object containing all punches in one object, seperated by a individual punch_id column

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

def train_test_split(dataFrame, predictionColumn, trainsize, seed=5):
    """ Splits the given dataset in train and test data for the given prediction target column.
        Returns a list containing the following objects:
            list[0]     --      Training dataset (type: pandas.core.frame.DataFrame)
            list[1]     --      Test dataset  (type: pandas.core.frame.DataFrame)
            list[2]     --      y_train target labels (type: pandas.core.series.Series)
            list[3]     --      y_test target labels  (type: pandas.core.series.Series)

    Keyword arguments:
    dataFrame                -- Dataset to train/test split (type: pandas.DataFrame)
    predictionColumn         -- The target column to predict (type: string)
    trainsize                -- Percentage of training data to create, e.g. 0.7 (type: number)
    seed                     -- Settings for reproduce the random process at multiple tests (type: number, default: 5)

    Returns:
        list
    """
    result = False
    if predictionColumn == 'label':
        punchIdx = dataFrame['punch_id'].unique()
        random.seed(seed)
        data = dataFrame.copy()
        punchIdx = data['punch_id'].unique()
        list_in_cpy = punchIdx[:]
        random.shuffle(list_in_cpy)
        n = len(list_in_cpy)
        idx_train = list_in_cpy[:round(n*trainsize)]
        idx_test = list_in_cpy[round(n*trainsize):]
        train_ds = data[data['punch_id'].isin(idx_train)]
        test_ds = data[data['punch_id'].isin(idx_test)]

        test_dataset_unique_label_id = test_ds.drop_duplicates(
            subset='punch_id', keep='first', inplace=False)
        y_test = pd.Series(data=test_dataset_unique_label_id[predictionColumn])
        train_dataset_unique_label_id = train_ds.drop_duplicates(
            subset='punch_id', keep='first', inplace=False)
        y_train = pd.Series(
            data=train_dataset_unique_label_id[predictionColumn])
        result = [train_ds.reset_index(drop=True), test_ds.reset_index(
            drop=True), y_train, y_test]
    else:
        print(
            'Error: Chosen predictionColumn not valid! Accepted: label, annotator or hand.')
    return result

def get_train_test_ratio(trainDataSet, testDataSet, column, decimals=1):
    """ Prints the ratio information of the training and test datasets for the given target column.

    Keyword arguments:
    trainDataSet             -- Train Dataset (type: pandas.core.frame.DataFrame)
    testDataSet              -- Test Dataset (type: pandas.core.frame.DataFrame)
    column                   -- The target column to predict (type: string, e.g.: 'label')
    decimals                 -- Number of decimals to print for the percentage values (type: number, default: 1)

    Returns:
        void
    """

    sumOfTrainData = 0
    print('------------------------')
    print('Training Dataset ratio:')
    for val in trainDataSet[column].value_counts():
        sumOfTrainData += val
    for idx, val in enumerate(trainDataSet[column].value_counts()):
        print("{} {}: {} %.".format(column, idx, round(
            ((val*100)/sumOfTrainData), decimals)))
    print('------------------------\n')
    print('Test Dataset ratio:')
    sumOfTestData = 0
    for val in testDataSet[column].value_counts():
        sumOfTestData += val
    for idx, val in enumerate(testDataSet[column].value_counts()):
        print("{} {}: {} %.".format(column, idx, round(
            ((val*100)/sumOfTestData), decimals)))
    print('------------------------\n')
    print('Ratio of training and test data:')
    print('Training data absolute: {} , Training data percentage of all: {} %'.format(len(trainDataSet['punch_id'].unique()), round((len(
        trainDataSet['punch_id'].unique())*100/(len(trainDataSet['punch_id'].unique())+len(testDataSet['punch_id'].unique()))), decimals)))
    print('Test data absolute:     {} , Test data percentage of all: {} %'.format(len(testDataSet['punch_id'].unique()), round((len(
        testDataSet['punch_id'].unique())*100/(len(trainDataSet['punch_id'].unique())+len(testDataSet['punch_id'].unique()))), decimals)))
    print('------------------------\n')

def extractFeatures(dataSetToExtractFrom, feature_settings="minimal"):
    """ Extracts features of the given dataset and returns a new dataset of features only.

    Keyword arguments:
    dataSetToExtractFrom     -- Dataset (type: pandas.core.frame.DataFrame)
    feature_settings         -- Feature extraction parameter (type: string, options: 'minimal','maximal', 'findBest')

    Returns:
        pandas.core.frame.DataFrame
    """

    dataset_for_extraction = dataSetToExtractFrom.drop(
        columns=['label'])

    if feature_settings == "minimal":
        extractedFeatures = MinimalFCParameters()
    elif feature_settings == "maximal":
        extractedFeatures = ComprehensiveFCParameters()
    elif feature_settings == "findBest":
        extractedFeatures = EfficientFCParameters()
    else:
        extractedFeatures = MinimalFCParameters()
        print('Given value for feature_parameter not valid! Minimal feature set is used instead.')

    extracted_featureset = extract_features(dataset_for_extraction, column_id="punch_id",
                                            column_sort="timestamp", impute_function=impute, default_fc_parameters=extractedFeatures)
    return extracted_featureset

def get_available_classifier_labels():
    """ Returns a list containing the names of the available ML classifier.

    Returns:
        list
    """

    return ['Linear SVC (ovr)', 'Standard SVC', 'Logsitic Regression', 'KNN', 'Random Forest']


def find_optimal_KNN_neighbors(k_Start, k_End, X_train, y_train, X_test, y_test):
    """ Finds the best parameter for k iteratively and returns it.

    Keyword arguments:
    k_Start                  -- Start value
    k_End                    -- End value
    X_train                  -- Training Dataset (type: pandas.core.frame.DataFrame)
    y_train                  -- Solutions of the training Dataset (type: pandas.core.frame.DataFrame)
    X_test                   -- Test Dataset (type: pandas.core.frame.DataFrame)
    y_test                   -- Solutions of the test Dataset (type: pandas.core.frame.DataFrame)

    Returns:
        number
    """

    error_rate = []
    for k in range(k_Start, k_End):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    print('KNN-classifier: Found optimal value for K-nearest at: {} with a error rate of: {}'.format(
        error_rate.index(min(error_rate))+1, (min(error_rate))))
    return error_rate.index(min(error_rate))+1


def predict(X_train, y_train, X_test, y_test, targetType, estimators=100, KNNneighbors='auto', KNN_Start=1, KNN_End=20, report=False, targetNames=False, exportModels=False):
    """ Creates the ML Models, trains and tests them. The function can find the best parameter for k iteratively. The created models can be exported.

    Keyword arguments:
    X_train                  -- Training Dataset (type: pandas.core.frame.DataFrame)
    y_train                  -- Solutions of the training Dataset (type: pandas.core.frame.DataFrame)
    X_test                   -- Test Dataset (type: pandas.core.frame.DataFrame)
    y_test                   -- Solutions of the test Dataset (type: pandas.core.frame.DataFrame)
    targetType               -- Target column to predict (type: string,e.g.: 'hand')
    estimators               -- Estimator value for RandomForest classifier (type: number,default: 100)
    KNNneighbors             -- k-Parameter for KNN classifier (type: number, default: 'auto')
    KNN_Start                -- If 'KNNneighbors' argument not 'auto': KNN_Start defines the start value for the search method (type: number, default: 1)
    KNN_End                  -- If 'KNNneighbors' argument not 'auto': KNN_End defines the end value for the search method (type: number, default: 20)
    report                   -- If True, the argument 'targetNames' needs to contain a list of strings representing the names of the target classes (type: boolean, default: False)
    targetNames              -- If 'report' argument True, the targetNames represents the names of the target classes as a list of strings (type: list)
    exportModels             -- If True, the created models gets exported to the filesystem as pickle files (type: boolean, default: False)

    Returns:
        list                 -- First list element is accuracy score. Second list element is a list object containing the created models
    """

    # inner helper function
    def print_classification_report(y_truth, y_predicted, target_labels):
        print(classification_report(y_true=y_truth,
                                    y_pred=y_predicted, target_names=target_labels))
        print(confusion_matrix(y_true=y_truth, y_pred=y_predicted))

    accuracy_scores = np.zeros(len(get_available_classifier_labels()))
    clf_labels = get_available_classifier_labels()
    # Linear Support Vector classifier
    linSupp_Vectr_clf = svm.LinearSVC()
    linSupp_Vectr_clf.fit(X_train, y_train)
    prediction = linSupp_Vectr_clf.predict(X_test)
    accuracy_scores[0] = accuracy_score(y_test, prediction)*100
    print(
        'Linear Vector Classifier accuracy (one-vs-rest): {}%'.format(accuracy_scores[0]))
    if report and targetNames:
        print('Classification report for {} model with target: {}'.format(
            'Linear-Vector-Classifier', targetType))
        print_classification_report(
            y_test, prediction, target_labels=targetNames)
    elif report and not targetNames:
        print('Cant create classification report because target names are missing or not valid!')
    else:
        print('Create next classifier model...')
    # Support Vector Classifier
    stdSupp_Vectr_clf = SVC().fit(X_train, y_train)
    prediction = stdSupp_Vectr_clf.predict(X_test)
    accuracy_scores[1] = accuracy_score(y_test, prediction)*100
    print('Support Vector Classifier accuracy: {}%'.format(accuracy_scores[1]))
    if report and targetNames:
        print('Classification report for {} model with target: {}'.format(
            'Support-Vector-Classifier', targetType))
        print_classification_report(
            y_test, prediction, target_labels=targetNames)
    elif report and not targetNames:
        print('Cant create classification report because target names are missing or not valid!')
    else:
        print('Create next classifier model...')
    # Logistic Regression
    logistic_reggr_clf = LogisticRegression().fit(X_train, y_train)
    prediction = logistic_reggr_clf.predict(X_test)
    accuracy_scores[2] = accuracy_score(y_test, prediction)*100
    print('Logistic Regression accuracy: {}%'.format(accuracy_scores[2]))
    if report and targetNames:
        print('Classification report for {} model with target: {}'.format(
            'Logistic Regression Classifier', targetType))
        print_classification_report(
            y_test, prediction, target_labels=targetNames)
    elif report and not targetNames:
        print('Cant create classification report because target names are missing or not valid!')
    else:
        print('Create next classifier model...')
    # K Nearest Neighbors
    # find optimal k parameter using elbow method
    if KNNneighbors == 'auto':
        K_value_to_choose = find_optimal_KNN_neighbors(
            KNN_Start, KNN_End, X_train, y_train, X_test, y_test)
    else:
        K_value_to_choose = KNNneighbors
    knn_clf = KNeighborsClassifier(
        n_neighbors=K_value_to_choose).fit(X_train, y_train)
    prediction = knn_clf.predict(X_test)
    accuracy_scores[3] = accuracy_score(y_test, prediction)*100
    print('K Nearest Neighbors (with {} neighbors value) Classifier accuracy: {}%'.format(
        K_value_to_choose, accuracy_scores[3]))
    if report and targetNames:
        print('Classification report for {} model with target: {}'.format(
            'KNN-Classifier', targetType))
        print_classification_report(
            y_test, prediction, target_labels=targetNames)
    elif report and not targetNames:
        print('Cant create classification report because target names are missing or not valid!')
    else:
        print('Create next classifier model...')
    # Random Forest
    rndm_forest_clf = RandomForestClassifier(
        n_estimators=estimators).fit(X_train, y_train)
    prediction = rndm_forest_clf.predict(X_test)
    accuracy_scores[4] = accuracy_score(y_test, prediction)*100
    print('Random Forest Classifier accuracy: {}%'.format(accuracy_scores[4]))
    if report and targetNames:
        print('Classification report for {} model with target: {}'.format(
            'Random Forest Classifier', targetType))
        print_classification_report(
            y_test, prediction, target_labels=targetNames)
    elif report and not targetNames:
        print('Cant create classification report because target names are missing or not valid!')
    else:
        print('Create next classifier model...')

    if exportModels:
        print("Start exporting the models as pickle files...")
        model_filename = 'linSupp_Vectr_clf'+'_'+targetType+'.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(linSupp_Vectr_clf, file)

        model_filename = 'stdSupp_Vectr_clf'+'_'+targetType+'.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(stdSupp_Vectr_clf, file)

        model_filename = 'logistic_reggr_clf'+'_'+targetType+'.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(logistic_reggr_clf, file)

        model_filename = 'knn_clf'+'_'+targetType+'.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(knn_clf, file)

        model_filename = 'rndm_forest_clf'+'_'+targetType+'.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(rndm_forest_clf, file)
        print("Models successfully exported to filesystem!")

    return [accuracy_scores, [linSupp_Vectr_clf, stdSupp_Vectr_clf, logistic_reggr_clf, knn_clf, rndm_forest_clf]]


def print_prediction_results(classifier_labels, acc_scores, plotTitle, xLabel='Classifiers', yLabel='Accuracy', figSizeTupel=(20, 10), saveImage=False, imageName='unknown_image.png'):
    """ Prints the prediction result of the tested models. Can optional export the plot to the filesystem.

    Keyword arguments:
    classifier_labels        -- Labels for the classifiers to plot (type: list)
    acc_scores               -- The classifiers accuracies (type: numpy.ndarray)
    plotTitle                -- Plot title (type: string)
    xLabel                   -- Label for the plots x axis (type: string, default: 'Classifiers')
    yLabel                   -- Label for the plots y axis (type: string, default: 'Accuracy')
    figSizeTupel             -- Figure size as optional parameter (type: tupel, default: (20,10))
    saveImage                -- If True the image is saved on the filesystem (type: boolean, default: False)
    imageName                -- If saveImage is True, the image is saved using the given imageName (type: string, default: "unknown_image.png")

    Returns:
        void
    """

    colors = cm.rainbow(np.linspace(0, 1, len(classifier_labels)))
    plt.figure(figsize=figSizeTupel)
    plt.bar(classifier_labels,
            acc_scores,
            color=colors)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(plotTitle)
    if saveImage:
        if imageName.endswith('.png'):
            plt.savefig(imageName)
        else:
            if "." not in imageName:
                name = imageName+'.png'
                plt.savefig(name)
            else:
                print("Error: File extension {} not allowed! See docs for more.".format(
                    imageName.split(".")[1]))


def load_model(pathToModel):
    """ Loads a Model as a pickle file from filesystem.

    Keyword arguments:
    pathToModel              -- Path to the models pickle file (type: string)

    Returns:
        object
    """
    theModel = pickle.load(open(pathToModel, "rb"))
    return theModel
