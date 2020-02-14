import os

import pandas as pd

"""
zastanowić się czy nie należy dodać klasy która będzie generyczna i będzie w stanie przyjmowac dane w określonym 
formacie, aby dalej umożliwić na nich prace reszty oprogramowania
"""
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