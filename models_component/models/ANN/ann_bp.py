import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

from models_component.models.ANN.ann import NeuralNetwork


class NeuralNetworkBP:

    def __init__(self):
        pass

    # =================================================================================================================
    # GROMADZENIE METRYK
    # =================================================================================================================

    # metody gromadzenia metryk

    # =================================================================================================================
    # PROCES UCZENIA
    # =================================================================================================================

    # aktualizacja wag

    # wsteczna propagacja błędu

    # test

    # train
    def train(self, train_set, model, model_config):
        pass

    # =================================================================================================================
    # METODY WALIDACJI
    # =================================================================================================================

    # walidacja za pomocą zbioru testowego
    def split_test_train(self, data, test_set_size):
        train_set, test_set = train_test_split(data, test_size=test_set_size)

        return train_set, test_set

    def simple_split(self, model, data, model_config):
        train_set, test_set = self.split_test_train(data, model_config['validation_mode']['test_set_size'])
        self.train(train_set, model, model_config)

    # walidacja krzyżowa

    # =================================================================================================================
    # METODY STERUJĄCE MODELEM
    # =================================================================================================================

    def init_model(self, data, model_config):
        n_inputs = len(data.loc[0]) - 1
        n_hidden = model_config['n_hidden']
        n_outputs = len(data.iloc[:, -1].unique())

        model = NeuralNetwork(n_inputs, n_hidden, n_outputs)
        model.create_network()

        return model.network

    def run(self, data, model_config):
        print("Start backpropagation")
        validation_method ={'simple_split':        self.simple_split}

        model = self.init_model(data, model_config)

        validation_method[model_config['validation_mode']['mode']](model, data, model_config)


        pass