import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def extract_accelerometer_data(csv_file_path, startpunkt=0,  nano=True, endpunkt=0, rechts=True):
    def invert_data(data):
        data['x'] = data['x'] * (-1)
        data['y'] = data['y'] * (-1)
        data['z'] = data['z'] * (-1)
        return data

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    if endpunkt != 0:
        df = df[startpunkt:endpunkt].copy()
    else:
        df = df[startpunkt:].copy()

    # Extract the accelerometer data columns
    # Modify the column names based on your file's actual headers
    accelerometer_data = df[
        ['accelerometerTimestamp_sinceReboot(s)', 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)',
         'accelerometerAccelerationZ(G)']]
    accelerometer_data = accelerometer_data.rename(
        columns={'accelerometerTimestamp_sinceReboot(s)': 'timestamp', 'accelerometerAccelerationX(G)': 'x',
                 'accelerometerAccelerationY(G)': 'y', 'accelerometerAccelerationZ(G)': 'z'})
    # drop rows with NaN values
    accelerometer_data = accelerometer_data.dropna()
    accelerometer_data = accelerometer_data.reset_index(drop=True)

    if rechts:
        accelerometer_data = invert_data(accelerometer_data)

    # let timestampt column start with 0 and change it from seconds to nanoseconds
    first_timestamp = accelerometer_data['timestamp'][0]
    accelerometer_data['timestamp'] = accelerometer_data['timestamp'] - first_timestamp
    if nano:
        accelerometer_data['timestamp'] = (accelerometer_data['timestamp'] * 1e9)

    # accelerometer_data['timestamp'] = (accelerometer_data['timestamp'] * 1000)

    # accelerometer_data['timestamp'] = (accelerometer_data['timestamp'] * 1e9).astype(float)

    # accelerometer_data.to_json(r'C:\Users\Raoul\Documents\GitHub\Boxschlag-Klassifikation\accelerometer_data.json', orient='records', lines=True)

    # returns a dataframe
    return accelerometer_data

def extract_gyroscope_data(csv_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the gyroscope data columns
    #gyroscope_data = df[['gyroscopeTimestamp_sinceReboot(s)', 'gyroscopeRotationX(rad/s)', 'gyroscopeRotationY(rad/s)', 'gyroscopeRotationZ(rad/s)']]
    gyroscope_data = df[['motionTimestamp_sinceReboot(s)', 'motionRotationRateX(rad/s)', 'motionRotationRateY(rad/s)', 'motionRotationRateZ(rad/s)']]
    gyroscope_data = gyroscope_data.rename(columns={'motionTimestamp_sinceReboot(s)':'timestamp', 'motionRotationRateX(rad/s)':'x', 'motionRotationRateY(rad/s)':'y', 'motionRotationRateZ(rad/s)':'z'})
    gyroscope_data['label'] = 0

    # drop rows with NaN values and reset index
    gyroscope_data = gyroscope_data.dropna()
    gyroscope_data = gyroscope_data.reset_index(drop=True)

    # let timestampt column start with 0 and change it from seconds to nanoseconds
    first_timestamp = gyroscope_data['timestamp'][0]
    gyroscope_data['timestamp'] = gyroscope_data['timestamp'] - first_timestamp
    gyroscope_data['timestamp'] = (gyroscope_data['timestamp'] * 1e9).astype(int)

    # Save the extracted gyroscope data to a new CSV file
    #gyroscope_data.to_json('gyroscope_data.json', orient='index')

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
        raw = {'_id': idx, 'timestamp': row['timestamp'], 'x': row['x'], 'y': row['y'], 'z': row['z']}
        raws.append(raw)

    # insert raws into dataset
    one_punch = {'label': label, 'count': len(data), 'periodNS': periodNS, 'raws': raws}

    return one_punch


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

    return ds_interp

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
        raw = {'_id': idx, 'timestamp': row['timestamp'], 'x': row['x'], 'y': row['y'], 'z': row['z']}
        raws.append(raw)

    # insert raws into dataset
    one_punch = {'label': label, 'count': len(data), 'periodNS': periodNS, 'raws': raws}

    return one_punch

def auto_labeling(data, height, distance, label):
    """
    Args:


    Returns:
    """
    # liste mit Schlägen
    punches = []

    # PCA
    r_data = reduce_dimensionality(data)

    # df mit den Hochpunkten
    p = scipy.signal.find_peaks(r_data['one_value'], height=height, distance=distance)
    peaks = data.iloc[p[0]]

    # eine Spalte mit den Indizes der Hochpunkte wird erstellt
    peaks = peaks.reset_index()
    print(peaks)

    # endpoint of punches
    end = 0

    first_iteration = True

    for idx, row in peaks.iterrows():
        # peak_index = row['index']
        if first_iteration:
            start = 0
            t = (peaks.loc[1, 'timestamp'] + peaks.loc[0, 'timestamp']) // 2
            t_2 = find_nearest_timestamp(data, t)
            end = find_index_by_timestamp(data, t_2)
            first_iteration = False
        elif idx == len(peaks) - 1:
            start = end + 1
            end = len(data) - 1
        else:
            # start = (peaks.loc[idx - 1, 'index'] + peaks.loc[idx, 'index']) // 2
            start = end + 1
            t = (peaks.loc[idx, 'timestamp'] + peaks.loc[idx + 1, 'timestamp']) // 2
            t_2 = find_nearest_timestamp(data, t)
            end = find_index_by_timestamp(data, t_2)

        print(str(idx) + '. Durchlauf: start:' + str(start) + ', ende:' + str(end))
        # print(peaks.loc[idx, 'index'])
        # Label the data from start to end
        ds = data_to_raw(data[start:end], label)
        punches.append(ds)

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
    print(index)
    return index

def reduce_dimensionality(data):
    r_data = data[['x', 'y', 'z']].copy()
    #time_c = data[['timestamp']].copy()
    pca = PCA(n_components=1)
    std_data = StandardScaler().fit_transform(r_data)
    gsts_array = pca.fit_transform(std_data)
    finalDf = pd.DataFrame(data = gsts_array, columns = ['one_value'])
    #finalDf = pd.concat([df,time_c], axis = 1)
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


def plot_ds(ds):
    x, y, z = [], [], []
    ds_ = ds.to_dict('records')

    for ds_e in ds_:
        x.append(ds_e['x'])
        y.append(ds_e['y'])
        z.append(ds_e['z'])

    t = np.arange(0, len(x))
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(t, x, label='x-Achse')
    ax.plot(t, y, label='y-Achse')
    ax.plot(t, z, label='z-Achse')

    ax.set(xlabel='datastamps', ylabel='a (in m/s²)',
           title='lecker Daten')
    ax.grid()
    plt.legend()
    plt.show()


def plot_ds_element(ds_e):
    raws = ds_e['raws']
    raws[0]
    x = data_to_list2(raws, axis='x')
    y = data_to_list2(raws, axis='y')
    z = data_to_list2(raws, axis='z')
    t = np.arange(0, len(x))

    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(t, x, label='x')
    ax.plot(t, y, label='y')
    ax.plot(t, z, label='z')

    ax.set(xlabel='datastamps', ylabel='a (in m/s²)',
           title='single dataset element of type: '+ds_e['label'])
    ax.grid()
    plt.legend()
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


def print_usage_info():
    print("use the functions in the notation of the following examples:")
    print(
        "universal_plotter_for_single_axis([ds[0],...,ds[n]],['axisToPrintForDs1','axisToPrintForDsN'],['ds[0] legend text','ds[N] legend text'],'Plot-Title','y-Axis description','y-Axis description')")
    print(
        "universal_plotter_for_all_axis([ds[0],...,ds[n]],['ds[0] legend text','ds[N] legend text'],'Plot-Title','y-Axis description','x-Axis description')")
    print("single_plot_single_axis(dataset,'axisToPrint','Plot-Title','y-Axis description','x-Axis description','default_axis_legend='legend text')")
    print("single_plot_all_axis(dataset,'Plot-Title','y-Axis description','x-Axis description','default_axis_legend='legend text')")

def merge_csv_files(file_paths):

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through the list of file paths and read each file into a DataFrame
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df

def csv_to_json(csv_file_path, json_file_path):
    # Read the CSV data
    df = pd.read_csv(csv_file_path)

    # Convert the CSV data to JSON
    json_data = df.to_json(orient='records')

    # Write the JSON data to a file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)
