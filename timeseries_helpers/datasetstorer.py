#!/usr/bin/env python
# coding: utf-8

import pandas as pd


def export_list_of_dataframes_to_csv(list_of_datasets, path):
    """Converts a given list of dataframes and converts it to a csv file to store on the given filesystem path.

    Keyword arguments:
        list_of_datasets        --  List object containing dataframe objects (type: list)
        path                    --  Filepath including the filename and extension for the csv file to export. Make shure to use the raw representation to prevent error messages (type: r"string")

    Returns:
        void
    """
    pd.concat(list_of_datasets).to_csv(path)
