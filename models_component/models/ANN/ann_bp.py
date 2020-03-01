import numpy as np
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

    def clear_metrics(self, metrics):
        """
        Metoda odpowiada za czyszczenie pól konfiguracji modelu, w których zbierane są metryki przed ich agregacją.
        Wykorzystywana jest przez metody, które uruchamiane są w sytuacji zapełnienia wspomnianych pól, a jednocześnie
        mają za zadanie gromadzenie metryk.
        :param metrics:
        :return:
        """
        metrics['n_epoch'] = []
        metrics['n_row'] = []
        metrics['prediction'] = []
        metrics['real_value'] = []

    def collect_metrics(self, n_epoch, n_row, prediction, real_value, metrics):
        """
        Zadaniem metody jest gromadzenie danych niezbędnych do obliczenia metryk jakości modelu i przypisanie ich do
        struktury danych obejmującej konfigurację modelu.
        :param n_epoch: int
        :param n_row: int
        :param prediction: list
        :param real_value: list
        :param metrics: dict
        :return:
        """
        metrics['n_epoch'].append(n_epoch)
        metrics['n_row'].append(n_row)
        metrics['prediction'].append(prediction)
        metrics['real_value'].append(real_value)

    def aggregate_metrics(self, model_config, data_target):
        """
        Metoda odpowiedzialna za agregacje zgromadzonych metryk:
        1. Budowany jest DataFrame, w którym umieszczane są kolumny odpowiednio dla numeru, epoki, wiersza, wartości
        predykcji i prawidłowej odpowiedzi w postacji konkretnych klas, oraz w rozbiciu dla każdej z nich;
        2. Wartości predykcji i prawidłowych odpowiedzi przetwarzane są do postaci zbiorczej i umieszczane
        w odpowiednich listach;
        3. Tworzona jest zbiorcza lista w której umieszczane są zgromadzone dane;
        4. Tak przetworzone dane umieszczane są w przygotoawnym wcześniej DataFrame;
        5. Gotowy DataFrame umieszczany jest w odpowiednim polu w strukturze konfiguracji;
        :param model_config: dict
        :param data_target: string
        :return:
        """
        columns = ['n_epoch', 'n_row', 'prediction', 'real_value']
        [columns.append('p' + str(i)) for i in range(len(model_config['metrics']['prediction'][0]))]
        [columns.append('r' + str(i)) for i in range(len(model_config['metrics']['real_value'][0]))]

        metrics = pd.DataFrame(columns=columns)

        predictions = [pred.index((max(pred))) for pred in model_config['metrics']['prediction']]
        real_value = [real.index((max(real))) for real in model_config['metrics']['real_value']]

        all_metrics = [model_config['metrics']['n_epoch'], model_config['metrics']['n_row'], predictions, real_value]

        pred = np.array(model_config['metrics']['prediction'])
        real = np.array(model_config['metrics']['real_value'])

        [all_metrics.append(list(pred_col)) for pred_col in list(pred.transpose())]
        [all_metrics.append(list(real_col)) for real_col in list(real.transpose())]

        for idx, column in enumerate(columns):
            metrics[column] = all_metrics[idx]

        model_config['metrics'][data_target].append(metrics)

    # =================================================================================================================
    # WALIDACJA MODELU
    # =================================================================================================================

    def test(self, test_set, model, model_config):
        """
        Metoda odpowiadająca za walidacje modelu za pomocą przekazanego zestawu danych testowych. Przechodzi przez
        wszystkie wiersze zbioru testowego:
        1. Dla każdego wiersza wykonuje predykcję;
        2. Określana jest prawidłowa odpowiedź;
        3. Gromadzone są metryki;
        4. Gromadzone są wyniki predykcji;
        Po zakonczeniu wszystkihc predykcji, zgromadzone metryki są agregowane.
        :param test_set:
        :param model:
        :param model_config:
        :return:
        """
        answers = []
        self.clear_metrics(model_config['metrics'])

        for idx, row in test_set.iterrows():
            outputs = model.feed_forward(row[:-1])

            answer = [0 for i in range(len(model.network[-1]))]
            answer[int(row[-1:] - 1)] = 1

            self.collect_metrics(0, idx, outputs, answer, model_config['metrics'])

            answers.append(outputs)

        self.aggregate_metrics(model_config, 'data_test')

    # =================================================================================================================
    # PROCES UCZENIA
    # =================================================================================================================

    # aktualizacja wag
    def weight_update(self, model, inputs, l_rate):
        """
        Metoda odpowiedzialna za aktualizowanie wag sieci. Przechodzi przez wszystkie warstwy:
        1. Wejście dla każdej z warstw ustalane jest jako wyjście z warstwy kolejnej;
        2. Do każdej wagi w kazdym neuronie dodawana jest wartość l_rate * delta * wejście;
        3. Dobiasu dodawana jest wartość l_rate * delta
        :param model: list
        :param inputs: list
        :param l_rate: float
        :return:
        """
        for i in range(len(model)):
            if i != 0:
                inputs = [neuron['output'] for neuron in model[i - 1]]
            for neuron in model[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
        pass

    def transfer_derivative(self, output):
        """
        Metoda obliczania wartości delty dla poszczególnych neuronów, przy pomocy funkcji sigmoidalnej.
        :param output: float
        :return: float
        """
        return output * (1.0 - output)

    def error_propagate(self, model, answer):
        """
        Metoda odpowiedzialna za wsteczną propagację błędu. Przechodzi, w odwróconej kolejności przez wszystkie warstwy
        sieci neuronowej:
        1. Dla ostatniej warstwy obliczany jest błąd jako różnicza pomiędzy wartościami na wyjściu sieci
        a prawidłowymi odpowiedziami;
        2. Dla każdego neuronu obliczana jest delta;
        3. Dla neuronów w warstwach ukrytych błąd obliczany jest jako iloczyn wartość wagi na połączeniu z neuronem
        w warstwie wyzszej i wartości jego delty
        :param model:
        :param answer:
        :return:
        """
        for i in reversed(range(len(model))):
            layer = model[i]
            errors = []

            if i != len(model) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in model[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)

            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(answer[j] - neuron['output'])

            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    def train(self, train_set, model, model_config):
        """
        Głowna metoda odpoiwedzilan za proces uczeni sieci neuronowej. Iteruje przez określoną ilość epok, i w każdej
        epoce przez wszystkie wiersze w zbiorze treningowym. W każdej takiej iteracji:
        1. Model wykonuje obliczenia dla danego wiersza i zwraca wynik;
        2. Określana jest prawidłowa odpoiwedz i obliczana jest suma kwadratów błędów;
        3. Gromadzone sa metryki uczenia;
        4. Przeprowadzany jest proces wstecznej propagacji błędu;
        5. Aktualizowane sa wagi sieci neuronowej;
        Po zakończeniu procesu uczenia wywoływana jest metoda odpowiedzialna za agregację zgromadzonych metryk.
        :param train_set: Pandas DataFrame
        :param model: list
        :param model_config: dict
        :return:
        """
        for epoch in range(model_config['n_epoch']):
            error_sum = 0

            for idx, row in train_set.iterrows():
                outputs = model.feed_forward(row[:-1])

                answer = [0 for i in range(len(model.network[-1]))]
                answer[int(row[-1:] - 1)] = 1
                error_sum += sum([(answer[i] - outputs[i])**2 for i in range(len(answer))])

                self.collect_metrics(epoch, idx, outputs, answer, model_config['metrics'])

                self.error_propagate(model.network[1:], answer)
                self.weight_update(model.network[1:], row[:-1], model_config['l_rate'])

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, model_config['l_rate'], error_sum))

        self.aggregate_metrics(model_config, 'data_train')

    # =================================================================================================================
    # METODY WALIDACJI
    # =================================================================================================================

    # walidacja za pomocą zbioru testowego
    def split_test_train(self, data, test_set_size):
        """
        Metoda dzieli zbiór danych na treningowy i testowy, zgodnie z zadanym współczynnikiem podziału.
        :param data: Pandas DataFrame
        :param test_set_size: int
        :return: tarin_set (Pandas DataFrame), test_set (Pandas DataFrame)
        """
        train_set, test_set = train_test_split(data, test_size=test_set_size)

        return train_set, test_set

    def simple_split(self, data, model_config):
        """
        Metoda obsługuje proces uczenia z wykorzystaniem walidacji za pomocą zbioru testowego. W pierwszej kolejności
        inicjalizowany jest obiekt klasy NeuralNetwork. Nastepnie dane dzielone są na zbiór testowy i treningowy oraz
        uruchamiane są metody odpowiedzialne za procesu uczenia i weryfikacji modelu.
        :param data: Pandas DataFrame
        :param model_config: dict
        :return:
        """
        model = self.init_model(data, model_config)

        train_set, test_set = self.split_test_train(data, model_config['validation_mode']['test_set_size'])

        self.train(train_set, model, model_config)
        self.test(test_set, model, model_config)

    def cross_validation(self, data, model_config):
        """
        Metoda walidacji krzyżowej. Dane dzielone są na k zbiorów z których każdy kolejno pełni rolę zbioru testowego.
        W każdej iteracji:
        1. Tworzony jest obiekt klasy NeuralNetwork;
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
            model = self.init_model(data, model_config)
            print('============================================')
            print('k_fold: ', i + 1)
            print('============================================')

            train_set = data.loc[train_index]
            train_set = shuffle(train_set)
            self.train(train_set, model, model_config)

            test_set = data.loc[test_index]
            test_set = shuffle(test_set)
            self.test(test_set, model, model_config)

    # =================================================================================================================
    # METODY STERUJĄCE MODELEM
    # =================================================================================================================

    def init_model(self, data, model_config):
        """
        Metoda inicjalizująca sieć neuronową. Na podstawie charakterystyk danych oraz ustawień konfiguracji określa
        liczbę wejść i wyjść do sieci oraz liczbę warstw i neuronów ukrytych. Na tej podstawie tworzony jest obiekt
        klasy NeuralNetwork.
        :param data: Pandas DataFrame
        :param model_config: dict
        :return: list
        """
        n_inputs = len(data.loc[0]) - 1
        n_hidden = model_config['n_hidden']
        n_outputs = len(data.iloc[:, -1].unique())

        model = NeuralNetwork(n_inputs, n_hidden, n_outputs)
        model.create_network()

        return model

    def run(self, data, model_config):
        """
        Głowna funkcja sterująca działaniem modelu. Przyjmuje dane oraz konfigurację modelu, a następnie uruchamia
        procedurę uczenia, z wykorzystaniem odpowiedniej metody walidacji.
        :param data: Pandas DataFrame
        :param model_config: dict
        :return:
        """
        print("Start backpropagation")
        validation_method ={'simple_split':         self.simple_split,
                            'cross_validation':     self.cross_validation}

        validation_method[model_config['validation_mode']['mode']](data, model_config)
