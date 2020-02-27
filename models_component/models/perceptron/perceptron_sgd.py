import random

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

from models_component.models.perceptron.perceptron import Perceptron


class PerceptronSGD:

    def __init__(self):
        pass

    # =================================================================================================================
    # GROMADZENIE METRYK
    # =================================================================================================================

    def clear_metrics(self, metrics):
        """
        Metoda odpowiada za czyszczenie pól konfiguracji modelu, w których zbierane są metryki przed ich agregacją.
        Wykorzystywana jest przez metody, które uruchamiane są w sytuacji zapełnienia wspomnianych pól, a jednocześnie
        mają za zadanie gromadzenie metryk.
        :param metrics: dict
        :return:
        """
        metrics['n_epoch'] = []
        metrics['n_row'] = []
        metrics['prediction'] = []
        metrics['real_value'] = []
        metrics['error'] = []

    def collect_metrics(self, n_epoch, n_row, prediction, real_value, error, metrics):
        """
        Zadaniem metody jest gromadzenie danych niezbędnych do obliczenia metryk jakości modelu i przypisanie ich do
        struktury danych obejmującej konfigurację modelu.
        :param n_epoch: int
        :param n_row: int
        :param prediction: int
        :param real_value: float
        :param error: float
        :param metrics: dict
        :return:
        """
        metrics['n_epoch'].append(n_epoch)
        metrics['n_row'].append(n_row)
        metrics['prediction'].append(prediction)
        metrics['real_value'].append(real_value)
        metrics['error'].append(error)

    def aggregate_metrics(self, model_config, data_target):
        """
        Metoda odpowiada za agregację metryk w zbiorczym polu konfiguracji modelu.
        :param model_config: dict
        :param data_target: string
        :return:
        """
        metrics = pd.DataFrame(columns=['n_epoch', 'n_row', 'prediction', 'real_value', 'error'])

        metrics['n_epoch'] = model_config['metrics']['n_epoch']
        metrics['n_row'] = model_config['metrics']['n_row']
        metrics['prediction'] = model_config['metrics']['prediction']
        metrics['real_value'] = model_config['metrics']['real_value']
        metrics['error'] = model_config['metrics']['error']

        model_config['metrics'][data_target].append(metrics)

    # =================================================================================================================
    # METODY UCZĄCE I TESTOWE
    # =================================================================================================================

    def test(self, perceptron, test_set, model_config):
        """
        Metoda walidacji na testowych danych. Przeprowadza ona predykcję na zbiorze danych testowych. Dla kazdego
        wiersza danych przeprowadzana jest predykcja oraz wywoływana metoda get_metrics.
        :param perceptron: obiekt klasy PerceptronSGDCore
        :param test_set: Pandas DataFrame
        :param model_config: dict
        :return: metrics (dict)
        """
        self.clear_metrics(model_config['metrics'])
        error_sum = 0

        for idx, row in test_set.iterrows():
            prediction = perceptron.predict(row[:1])

            error = row.iloc[-1] - prediction
            error_sum += error**2

            self.collect_metrics(0, idx, prediction, row.iloc[-1], error, model_config["metrics"])

            print('row: ', idx, ', error: ', error, ', error_sum: ', error_sum)

        self.aggregate_metrics(model_config, 'data_test')

    def train(self, perceptron, train_set, model_config):
        """
        Głowna metoda odpowiedzialna za przeprowadzenie procesu uczenia. Ilość iteracji określona jest za pomocą
        wartości pola model_config['n_epoch']. W każdej iteracji suma błędów inicjalizowana jest wartością 0,
        następnie dla każdego z wierszy przeprowadzana jest predykcja, obliczany jest błąd, a kwadrat jego wartości
        dodawany jest do sumy błędów. Po wykonaniu predykcji i obliczeniu błędu, dla każdego z wierszy aktualizowane są
        wagi oraz zbierane są metryki.
        :param perceptron: obiekt klasy PerceptronSGDCore
        :param train_set: Pandas DataFrame
        :param model_config: dict
        :return: metrics (dict)
        """
        print('Proces uczenia')
        for epoch in range(model_config['n_epoch']):
            error_sum = 0

            for idx, row in train_set.iterrows():
                prediction = perceptron.predict(row[:-1])
                error = row.iloc[-1] - prediction
                error_sum += error ** 2

                self.collect_metrics(epoch, idx, prediction, row.iloc[-1], error, model_config["metrics"])

                perceptron.weights[0] += model_config['l_rate'] * error

                for idx, x in enumerate(row[:-1]):
                    perceptron.weights[idx + 1] += model_config['l_rate'] * error * x

            print('epoch: ', epoch, 'error: ', error_sum)

        self.aggregate_metrics(model_config, 'data_train')

    # =================================================================================================================
    # WALIDACJA ZA POMOCĄ ZBIORU TESTOWEGO
    # =================================================================================================================

    def split_test_train(self, data, test_set_size):
        """
        Metoda dzieli zbiór danych na treningowy i testowy, zgodnie z zadanym współczynnikiem podziału.
        :param data: Pandas DataFrame
        :param test_set_size: int
        :return: tarin_set (Pandas DataFrame), test_set (Pandas DataFrame)
        """
        train_set, test_set = train_test_split(data, test_size=test_set_size)

        return train_set, test_set

    def simple_split(self, data, model_config): # dodać doc stringa
        """
        Metoda obsługuje proces uczenia z wykorzystaniem walidacji za pomocą zbioru testowego. W pierwszej kolejności
        inicjalizowany jest obiekt klasy PerceptronSGDCore. Nastepnie dane dzielone są na zbiór testowy i treningowy
        oraz uruchamiane są metody odpowiedzialne za procesu uczenia i weryfikacji modelu.
        :param data: Pandas DataFrame
        :param model_config: dict
        :return:
        """
        perceptron = Perceptron(len(data.iloc[0][:-1]))
        train_set, test_set = self.split_test_train(data, model_config['validation_mode']['test_set_size'])

        self.train(perceptron, train_set, model_config)
        self.test(perceptron, test_set, model_config)

    # =================================================================================================================
    # WALIDACJA ZA POMOCĄ METODY KRZYŻOWEJ
    # =================================================================================================================

    def cross_validation(self, data, model_config):
        """
        Proces uczenia z wykorzystaniem walidacji krzyżowej. Dane dzielone są na k zbiorów z których każdy kolejno
        pełni rolę zbioru testowego. W każdej iteracji:
        1. Tworzony jest obiekt klasy PerceptronSGDCore;
        2. Wybrane dane są przydzielane do zbioru treningowego, który jest następnie mieszany;
        3. Uruchamiana jest procedura uczenia;
        4. Przydzielane są dane do zbioru testowego, który jest następnie mieszany;
        5. Uruchamiana jest procedura testowania modelu;
        :param data: Pandas DataFrame
        :param model_config:
        :return:
        """
        kf = KFold(n_splits=model_config['validation_mode']['k'])

        for i, (train_index, test_index) in enumerate(kf.split(data)):
            perceptron = Perceptron(len(data.iloc[0][:-1]))
            print('============================================')
            print('k_fold: ', i+1)
            print('============================================')

            train_set = data.loc[train_index]
            train_set = shuffle(train_set)
            self.train(perceptron, train_set, model_config)

            test_set = data.loc[test_index]
            test_set = shuffle(test_set)
            self.test(perceptron, test_set, model_config)

    # =================================================================================================================
    # METODY STERUJĄCE MODELEM
    # =================================================================================================================

    def run(self, data, model_config):
        """
        Głowna funkcja sterująca działaniem perceptronu. Przyjmuje dane oraz konfigurację modelu, a następnie uruchamia
        procedurę uczenia, z wykorzystaniem odpowiedniej metody walidacji.
        :param data: Pandas DataFrame
        :param model_config: dict
        :return: results (list?), metrics (dict)
        """
        print('Start perceptron SGD')
        validation_method = {'simple_split':        self.simple_split,
                             'cross_validation':    self.cross_validation}

        validation_method[model_config['validation_mode']['mode']](data, model_config)
