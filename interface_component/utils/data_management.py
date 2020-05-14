import base64
import io
import os

import pandas as pd

from models_component.utils.data_preprocessing import DataPreprocessing


class DataManagement:

    def __init__(self):
        self.data = None
        self.datasets = []
        self.preprocessing = DataPreprocessing()
        pass

    def get_data(self):
        return self.data

    def read_data(self, data):
        try:
            data_type, data_string = data.split(',')

            data_decoded = base64.b64decode(data_string)

            self.data = pd.read_csv(io.StringIO(data_decoded.decode('utf-8')), header=None)

            columns = ['Cloumn ' + str(i) for i in range(len(self.data.columns) - 1)]

            columns.append('Answer')

            self.data.columns = columns

            return True

        except:
            return False

    def data_normalization(self):
        norm_data = self.preprocessing.data_normalization(self.data)

        columns = ['Cloumn ' + str(i) for i in range(len(self.data.columns) - 1)]

        columns.append('Answer')

        norm_data.columns = columns
        self.data = norm_data

    def label_encoding(self):
        label_data = self.preprocessing.label_encoding(self.data)
        self.data = label_data

    def shuffle_data(self):
        shuffle_data = self.data.sample(frac=1).reset_index(drop=True)
        self.data = shuffle_data

    def save_data(self, filename):
        filepath = os.path.join(os.getcwd() + '/models_component/data/' + filename + '.csv')

        self.data.to_json(filepath, index=False)

    def get_datasets_list(self):
        data_path = os.path.join(os.getcwd() + '/models_component/data/')

        datasets = os.listdir(data_path)

        if len(datasets) == 0:
            datasets = ['Brak danych.']

        return datasets