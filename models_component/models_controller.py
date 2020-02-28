import os

import pandas as pd

from models_component.models.perceptron.perceptron_sgd import PerceptronSGD
from models_component.models.perceptron.perceptron_ga import PerceptronGA
from models_component.utils.data_preprocessing import SonarDataPreprocessing
from models_component.utils.data_preprocessing import DataPreprocessing


class ModelController:

    def __init__(self):
        self.perceptron_sgd = PerceptronSGD()
        self.perceptron_ga = PerceptronGA()

        self.data_preprocess = {'sonar_data':   SonarDataPreprocessing(),
                                'seed_data':    DataPreprocessing()}
        self.data_source = {'sonar_data': os.path.join(os.getcwd() + '/models_component/data/sonar.all-data.csv')}

    def get_data(self, data_name):
        # sonar data
        # data = pd.read_csv(self.data_source[data_name])

        # self.data_preprocess[data_name].test_run()
        # data = self.data_preprocess[data_name].run_preprocessing()

        # seed data
        data = self.data_preprocess[data_name].run_preprocessing(data_name)
        return data

    def run_model(self, config):
        # print('number of epoch: ', model_config['n_epoch'])

        data = self.get_data(config['data_source'])    # data_source

        model = getattr(self, config['model'])
        model.run(data, config['model_config']) # wywalić results i metrics

        print()
