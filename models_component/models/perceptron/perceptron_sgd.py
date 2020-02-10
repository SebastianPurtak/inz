import random

from sklearn.model_selection import train_test_split


class PerceptronSGDCore:

    def __init__(self, n_inputs):
        """
        Inicjalizacja wag, jako tablicy złożonej z samych 0, o dłógości równej ilości wejść + bias.
        :param n_inputs:
        """
        self.weights = [random.uniform(0,1) for input in range(n_inputs + 1)]

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

    # metrics()

    # get_metrics()
    def mean_error_sum(self, error_sum, real_result, prediction):
        error = real_result - prediction
        error_sum += error_sum**2
        return error_sum

    def get_metric(self, real_result, prediction, error_sum, model_config):
        error_sum = self.mean_error_sum(error_sum, real_result, prediction)
        return error_sum

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
        error_sum = 0

        for idx, row in test_set.iterrows():
            prediction = perceptron.predict(row[:1])

            # wywołanie get_metrics
            error = prediction - row.iloc[-1]
            error_sum += error**2

            print('row: ', idx, ', error: ', error, ', error_sum: ', error_sum)


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
        for epoch in range(model_config['n_epoch']):
            error_sum = 0

            for idx, row in train_set.iterrows():
                prediction = perceptron.predict(row[:-1])
                error = row.iloc[-1] - prediction
                error_sum += error ** 2

                # error_sum = self.get_metric(row.iloc[-1], prediction, error_sum, model_config)

                perceptron.weights[0] += model_config['l_rate'] * error

                for idx, x in enumerate(row[:-1]):
                    perceptron.weights[idx + 1] += model_config['l_rate'] * error * x

            print('epoch: ', epoch, 'error: ', error_sum)


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
        validation_method = {'simple_split':    self.simple_split}

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