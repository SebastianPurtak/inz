import os

import pandas as pd
from sklearn.preprocessing import normalize

"""
zastanowić się czy nie należy dodać klasy która będzie generyczna i będzie w stanie przyjmowac dane w określonym 
formacie, aby dalej umożliwić na nich prace reszty oprogramowania
"""
class DataPreprocessing:

    def __init__(self):
        pass

    def get_data(self, data_name):
        path = os.path.join(os.getcwd() + '/models_component/data/' + data_name + '.csv')
        data = pd.read_csv(path, header=None)

        data.rename(columns={data.columns[-1]: 'Answer'}, inplace=True)

        return data

    def data_normalization(self, data):
        norm_data = pd.DataFrame(normalize(data.iloc[:, :-1]))
        norm_data['Answer'] = data.iloc[:, -1]

        return norm_data

    def run_preprocessing(self, data_name):
        data = self.get_data(data_name)
        data = self.data_normalization(data)

        return data


class SonarDataPreprocessing:

    def __init__(self):
        print()

    def get_data(self):
        # TODO: przemyśleć czy można rozwiązać dodawanie danych w sposób bardziej generyczny
        data_path = os.path.join(os.getcwd() + '/models_component/data/sonar.all-data.csv')
        columns = [nr for nr in range(60)]
        columns.append('Type')
        data = pd.read_csv(data_path, names=columns)
        return data

    def label_encoding(self, data):
        data['Type'] = data['Type'].astype('category')
        data['Type'] = data['Type'].cat.codes
        return data

    def run_preprocessing(self):
        data = self.get_data()
        data = self.label_encoding(data)
        return data


    def test_run(self):
        print('Preprocessing test run')