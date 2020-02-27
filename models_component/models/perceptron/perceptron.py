import random

class Perceptron:

    def __init__(self, n_inputs):
        """
        Inicjalizacja wag, jako tablicy złożonej z samych 0, o dłógości równej ilości wejść + bias.
        :param n_inputs:
        """
        # TODO: zastanowić się nad bardziej generycznym sposobem inicjalizowania wag
        self.weights = [random.uniform(-1,1) for input in range(n_inputs + 1)]

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

    def set_weights(self, new_weights):
        self.weights = new_weights