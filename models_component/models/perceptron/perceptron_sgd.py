import random

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle


class PerceptronSGDCore:

    def __init__(self, n_inputs):
        """
        Inicjalizacja wag, jako tablicy złożonej z samych 0, o dłógości równej ilości wejść + bias.
        :param n_inputs:
        """
        self.weights = [random.uniform(0,1) for input in range(n_inputs + 1)]
        # self.weights = [0 for input in range(n_inputs + 1)]

    def rectified_linear_activation_function(self, activation):
        """
        Unipolarna funkcja aktywacji.
        :param activation: float
        :return: wartość aktywacji (int)
        """
        if activation >= 0:
            return 1
        else:
            return 0

    def predict(self, data_row):
        """
        Metoda odpowiedzialna za przeprowadzenie predykcji. Przyjmuje jeden wiersz danych oraz oblicza wartość
        aktywacji dla wszystkich wejści.
        :param data_row: Pandas DataFrame
        :return: wartość aktywacji (int lub float)
        """
        activation = 0

        for i, input in enumerate(data_row):
            activation += self.weights[i + 1] * input

        activation += self.weights[0]

        return self.rectified_linear_activation_function(activation)


class PerceptronSGDUtils:

    def __init__(self):
        print()

    # GROMADZENIE METRYK

    def clear_metrics(self, metrics):
        metrics['n_epoch'] = []
        metrics['n_row'] = []
        metrics['prediction'] = []
        metrics['real_value'] = []
        metrics['error'] = []

    def collect_metrics(self, n_epoch, n_row, prediction, real_value, error, metrics):
        metrics['n_epoch'].append(n_epoch)
        metrics['n_row'].append(n_row)
        metrics['prediction'].append(prediction)
        metrics['real_value'].append(real_value)
        metrics['error'].append(error)

    def get_metrics(self, metrics_data, metrics, N):
        # TODO: upewnić się czy to jest potrzebne
        metrics['n_epoch'] = metrics_data['n_epoch']
        metrics['n_row'] = metrics_data['n_row']
        metrics['prediction'] = metrics_data['prediction']
        metrics['real_value'] = metrics_data['real_value']
        metrics['error'] = metrics_data['error']
        metrics['N'] = N

        return metrics

    # METODY UCZĄCE I TESTOWE

    # test()
    def test(self, perceptron, test_set, model_config):
        """
        Metoda walidacji na testowych danych. Przeprowadza ona predykcję na zbiorze danych testowych. Dla kazdego
        wiersza danych przeprowadzana jest predykcja oraz wywoływana metoda get_metrics.
        :param perceptron: obiekt klasy PerceptronSGDCore
        :param test_set: Pandas DataFrame
        :param model_config: dict
        :return: metrics (dict)
        """
        metrics = pd.DataFrame(columns=['n_row', 'prediction', 'real_value', 'error'])
        self.clear_metrics(model_config['metrics'])
        error_sum = 0

        for idx, row in test_set.iterrows():
            prediction = perceptron.predict(row[:1])

            # wywołanie get_metrics
            error = row.iloc[-1] - prediction
            error_sum += error**2

            self.collect_metrics(0, idx, prediction, row.iloc[-1], error, model_config["metrics"])

            print('row: ', idx, ', error: ', error, ', error_sum: ', error_sum)

        metrics = self.get_metrics(model_config['metrics'], metrics, len(test_set))
        # model_config['metrics']['data_test'] = metrics
        model_config['metrics']['data_test'].append(metrics)


    # train()
    def train(self, perceptron, train_set, model_config):
        """
        Głowna metoda odpowiedzialna za przeprowadzenie procesu uczenia. Ilość iteracji określona jest za pomocą
        wartości pola model_config['n_epoch']. W każdej iteracji suma błędów inicjalizowana jest wartością 0,
        następnie dla każdego z wierszy przeprowadzana jest predykcja, obliczany jest błąd, a kwadrat jego wartości
        dodawany jest do sumy błędów. Po wykonaniu predykcji i obliczeniu błędu, dla każdego z wierszy aktualizowane są
        wagi oraz uruchamiana jest metoda get_metrics.
        :param perceptron: obiekt klasy PerceptronSGDCore
        :param train_set: Pandas DataFrame
        :param model_config: dict
        :return: metrics (dict)
        """
        print('Proces uczenia')
        metrics = pd.DataFrame(columns=['n_epoch', 'n_row', 'prediction', 'real_value', 'error'])
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

        metrics = self.get_metrics(model_config['metrics'], metrics, len(train_set))
        # model_config['metrics']['data_train'] = metrics
        model_config['metrics']['data_train'].append(metrics)



    # WALIDACJA ZA POMOCĄ ZBIORU TESTOWEGO

    # train_test_split()
    def split_test_train(self, data, test_set_size):
        """
        Metoda dzieli zbiór danych na treningowy i testowy, zgodnie z zadanym wpsólczynnikiem podziału.
        :param data: Pandas DataFrame
        :param test_set_size: int
        :return: tarin_set (Pandas DataFrame), test_set (Pandas DataFrame)
        """
        train_set, test_set = train_test_split(data, test_size=test_set_size)

        return train_set, test_set

    # simple_split()
    def simple_split(self, data, model_config):
        perceptron = PerceptronSGDCore(len(data.iloc[0][:-1]))
        train_set, test_set = self.split_test_train(data, model_config['validation_mode']['test_set_size'])

        self.train(perceptron, train_set, model_config)
        self.test(perceptron, test_set, model_config)

        return 'results', 'metrics'

    # WALIDACJA ZA POMOCĄ METODY KRZYŻOWEJ

    # k_fold_split
    def k_fold_split(self, data, k):
        # data = shuffle(data)
        # data.reset_index()
        kf = KFold(n_splits=k)
        for train_index, test_index in kf.split(data):
            train_set = data.loc[train_index]
            train_set = shuffle(train_set)
            print()
        print()

    # cross_validation
    def cross_validation(self, data, model_config):
        # k_fold_data = self.k_fold_split(data, model_config['validation_mode']['k'])


        kf = KFold(n_splits=model_config['validation_mode']['k'])

        for i, (train_index, test_index) in enumerate(kf.split(data)):
            perceptron = PerceptronSGDCore(len(data.iloc[0][:-1]))
            print('k_fold: ', i+1)

            train_set = data.loc[train_index]
            train_set = shuffle(train_set)
            self.train(perceptron, train_set, model_config)

            test_set = data.loc[test_index]
            test_set = shuffle(test_set)
            self.test(perceptron, test_set, model_config)


        return 'x', 'y'


    # run()

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

        results, metrics = validation_method[model_config['validation_mode']['mode']](data, model_config)

        return 'results', 'metrics'



















    # Funkcje do dalszego przetworzenia danych

    # Funkcje do treningu
    # def train(self, data, n_epoch, l_rate, perceptron):
    #     for epoch in range(n_epoch):
    #         error_sum = 0
    #
    #         for idx, row in data.iterrows():
    #             prediction = perceptron.predict(row[:-1])
    #
    #             error = float(row[-1:] - prediction)
    #             error_sum += error**2
    #
    #             perceptron.weights[0] += l_rate * error
    #
    #             for i, x in enumerate(row[:-1]):
    #                 perceptron.weights[i + 1] += l_rate * error * x
    #         print('epoch: ', epoch, 'error: ', error_sum)

    # Ewentualna funkcja do testowania
    # def test(self, data, perceptron):
    #     good = 0
    #     bad = 0
    #     for idx, row in data.iterrows():
    #         prediction = perceptron.predict(row[:-1])
    #         print('prediction:', prediction, 'expected: ', int(row[-1:]))
    #         if prediction == int(row[-1:]):
    #            good += 1
    #         else:
    #             bad += 1
    #     print('Good prediction: ', good, 'Bad prediction: ', bad)

    # def run(self, data):
    #     print('Start perceptron SGD')
    #     perceptron = PerceptronSGDCore(len(data.iloc[0][:-1]))
    #
    #     data = data.sample(frac=1)
    #     train = data.iloc[:150, :]
    #     test = data.iloc[150:,:]
    #
    #     self.train(train, 50, 0.01, perceptron)
    #     self.test(test, perceptron)


        # data = self.model_controller.get_data()

        # print('data: ', data)