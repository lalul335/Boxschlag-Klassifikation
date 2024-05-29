#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
import json
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import sys

# helper functions
from timeseries_helpers import datasetstorer
from timeseries_helpers import database_importer
from timeseries_helpers import standardizer

# tsfresh modules (for feature extraction)
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
import logging
# Set logger-level to "error". Not recommed: Important warnings can be overseen
logging.basicConfig(level=logging.ERROR)


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
    ds_orig = database_importer.jsonData_to_dataset_in_timedifference_us(
        data=jsnDataset)
    ds_equalPeriod = standardizer.normate_dataset_period(
        periodLengthInMS, sampleRateInUS, ds_orig)
    ds = standardizer.change_strings_to_numbs(ds_equalPeriod)
    ds_equalPeriod = []
    ds_orig = []
    if exportToCSV:
        fileName = r"dataset_periodMS" + \
            str(periodLengthInMS)+"_sampleUS"+str(sampleRateInUS)+".csv"
        datasetstorer.export_list_of_dataframes_to_csv(ds, fileName)
    return ds


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
    if predictionColumn == 'label' or predictionColumn == 'annotator' or predictionColumn == 'hand':
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
        columns=['label', 'hand', 'annotator'])

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
