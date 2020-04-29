import os

from models_component.models.perceptron.perceptron_sgd import PerceptronSGD
from models_component.models.perceptron.perceptron_ga import PerceptronGA
from models_component.models.ANN.ann_bp import NeuralNetworkBP
from models_component.models.ANN.ann_ga import NeuralNetworkGA
from models_component.utils.data_preprocessing import DataPreprocessing


class ModelController:

    def __init__(self):
        self.perceptron_sgd = PerceptronSGD()
        self.perceptron_ga = PerceptronGA()
        self.ann_bp = NeuralNetworkBP()
        self.ann_ga = NeuralNetworkGA()

        self.data_source = {'sonar_data': os.path.join(os.getcwd() + '/models_component/data/sonar.all-data.csv')}

    def get_data(self, data_name):
        data_preprocessing = DataPreprocessing()
        data = data_preprocessing.get_data(data_name)
        return data

    def run_model(self, config):
        data = self.get_data(config['data_source'])    # data_source

        model = getattr(self, config['model'])
        model.run(data, config['model_config'])

        print()
