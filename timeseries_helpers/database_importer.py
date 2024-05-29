#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import datetime as datetime


def jsonData_to_dataset_in_time_format(data):
    """Creates a list of dataframe objects from a given json object. Converts the timestamp col in datetime object format.
       Converting mechanism is made for the notation style of the smartPunch project.
       If changes are made here make shure to also change the export/import function(s)!

    Keyword arguments:
        data                -- JSON Database representation dumped from mongoDB (type: object)

    Returns:
        list        --  List object containing the datasets as dataframe objects with timestamps in datetime object format.
    """

    the_cols = ['x', 'y', 'z', 'timestamp', 'label', 'hand', 'annotator']
    the_data = []

    for value in data:
        the_raws = []
        the_indxs = []
        idx = 0
        cur_time = datetime.datetime.strptime(
            value['createdAt']['$date'], '%Y-%m-%dT%H:%M:%S.%fZ')
        for raw in value['raws']:
            micro = int(raw['timestamp'])/1000
            raw_time = cur_time + datetime.timedelta(microseconds=micro)
            the_raws.append([raw['x'], raw['y'], raw['z'], int(
                raw_time), value['label'], value['hand'], value['annotator']])
            cur_time = raw_time
            the_indxs.append(idx)
            idx += 1
        the_data.append(pd.DataFrame(the_raws, the_indxs, the_cols))
    return the_data

# convert json data to dataframe list with timestamp in absolute us


def jsonData_to_dataset_in_timedifference_us(data):
    """Creates a list of dataframe objects from a given json object. Converts the timestamp col with absolute timestamps in us.
       The last timestamp is the period time in us since the punch started.

    Keyword arguments:
        data            -- JSON Database representation dumped from mongoDB with timestamps in nanoseconds (type: object)

    Returns:
        list            -- List object containing the datasets as dataframe objects with timestamps in 'since last timestamp' format.
    """

    the_cols = ['x', 'y', 'z', 'timestamp', 'label', 'hand', 'annotator']
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


def import_local_json_database_file(filename):
    """ Reads a json file from the file system and returns the json object representation of the file.
    Converting mechanism is made for the notation style of the smartPunch project.
    If changes are made here make shure to also change the export/import function(s)!

    Keyword arguments:
        filename           -- JSON file containing the file extension (type: string)

    Returns:
        object
    """

    # open json dataset
    with open(filename, 'r') as f:
        jsnDataset = json.load(f)
    return jsnDataset


def reduce_class_size(dataset, reductionType, targetSize):
    """Reduces the dataset size to save time. Use this for test purposes if not all punches are needed

    Keyword arguments:
        dataset             -- List of dataframe objects containing the punches (type: list)
        reductionType       -- Defines the type of the 'targetSize' argument (type: string, use: 'percentage' or 'absolute')
        targetSize          -- Defines the number of punches each class is reduced to. If there are not
                               as much punches in the original dataset of one class than defined in the argument, all punches of the class are imported (type: number, valid intervals: (0,1) or (0,100))

    Returns:
        list                -- List object containing the reduced dataset
    """

    classes = []
    classesContent = []
    ds_reduced = []
    originalDataset = dataset.copy()
    absOriginalLength = len(originalDataset)
    absTargetLength = 0
    targetMultiplicator = 0

    # calc absolute length to reduce to
    if (reductionType == 'percentage'):
        if (targetSize < 1 and targetSize > 0):
            targetMultiplicator = 100*targetSize
        elif (targetSize < 100 and targetSize > 0):
            targetMultiplicator = 1
        else:
            print("targetSize not valid! Use either a value less than one or a value less than 100 but always greater than 0")
            return
        absTargetLength = (absOriginalLength/100)*targetMultiplicator
    elif (reductionType == 'absolute') and (targetSize < absOriginalLength) and (targetSize > 0):
        absTargetLength = targetSize
    else:
        print("ReductionType or targetSize not valid! Use: percentage (value greater 0 and less than 1) or absolute (value greater 0 and less than dataset size)")
        return

    # find all available lable types
    for(ind, punch) in enumerate(originalDataset):
        if(punch.label[0] not in classes):
            classes.append(punch.label[0])
            classesContent.append(0)

    print("Found the following classes: {}".format(classes))

    # reduce the size of the dataset
    for (ind, punch) in enumerate(originalDataset):
        if(classesContent[classes.index(punch.label[0])] < absTargetLength):
            classesContent[classes.index(punch.label[0])] += 1
            ds_reduced.append(punch.copy())

    print("class balance: {}".format(classesContent))
    return ds_reduced
