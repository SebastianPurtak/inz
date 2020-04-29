import os

import pandas as pd
from sklearn.preprocessing import normalize


class DataPreprocessing:

    def __init__(self):
        pass

    def get_data(self, data_name):
        path = os.path.join(os.getcwd() + '/models_component/data/' + data_name)
        data = pd.read_csv(path, header=0)

        return data

    def data_normalization(self, data):
        norm_data = pd.DataFrame(normalize(data.iloc[:, :-1]))
        norm_data['Answer'] = data.iloc[:, -1]

        return norm_data

    def label_encoding(self, data):
        data['Answer'] = data['Answer'].astype('category')
        data['Answer'] = data['Answer'].cat.codes
        return data

    def run_preprocessing(self, data_name):
        data = self.get_data(data_name)
        data = self.data_normalization(data)

        return data