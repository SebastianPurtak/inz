import os

import pandas as pd

from models_component.models.perceptron.perceptron_sgd import PerceptronSGD
from models_component.utils.data_preprocessing import SonarDataPreprocessing


class ModelController:

    def __init__(self):
        self.perceptron_sgd = PerceptronSGD()

        self.data_preprocess = {'sonar_data': SonarDataPreprocessing()}
        self.data_source = {'sonar_data': os.path.join(os.getcwd() + '/models_component/data/sonar.all-data.csv')}

    def get_data(self, data_name):
        data = pd.read_csv(self.data_source[data_name])

        # self.data_preprocess[data_name].test_run()
        data = self.data_preprocess[data_name].run_preprocessing()
        return data

    def run_model(self, model_name, model_config):
        print('number of epoch: ', model_config['n_epoch'])

        data = self.get_data(model_config['data_name'])

        model = getattr(self, model_name)
        model.run(data)

        print()