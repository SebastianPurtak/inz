import random
from math import exp


class NeuralNetwork:
    """
    Podstawowa klasa warstwowej sieci neuronowej. Zawiera metody pozwalające na tworzenie modelu oraz wykonanie przez
    niego obliczeń bez uwzględniania procesu uczenia.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.network = []

    def sigmoid_activation_function(self, activation):
        """
        Sigmoidalna funkcja aktywacji.
        :param activation: float
        :return: float
        """
        return 1.0 / (1.0 + exp(-activation))

    def activate(self, inputs, neuron):
        """
        Metoda obliczająca wartośc aktywacji dla poszczególnych neuronów.
        :param inputs: float
        :param neuron: dict
        :return: float
        """
        activation = 0

        for idx, input in enumerate(inputs):
            activation += input * neuron['weights'][idx]

        activation += neuron['weights'][-1]

        return self.sigmoid_activation_function(activation)

    def set_weights(self, weights, genom_size):
        net = self.network[1:]

        for layer in net:
            for idx, neuron in enumerate(layer):
                no_weights = genom_size[0]
                del genom_size[0]
                neuron['weights'] = []

                for i in range(no_weights):
                    neuron['weights'].append(weights[0])
                    del weights[0]


        pass

    def feed_forward(self, inputs):
        """
        Metoda odpowiedzialna za wykonywanie obliczeń przez model. Przyjmuje określony sygnał, który wprowadza do
        pierwszej warstwy i stopniowo przeprowadza przez kolejne.
        :param inputs:
        :return: list
        """
        self.network[0] = inputs
        outputs = []

        for layer in self.network[1:]:
            outputs = []

            for neuron in layer:
                output = self.activate(inputs, neuron)
                outputs.append(output)
                neuron['output'] = output

            inputs = outputs

        return outputs

    def create_network(self):
        """
        Tworzy sieć jako zagnieżdżoną listę, w której każda podlista odpowiada jednej warstwie. Poszczególne neurony
        reprezentowane są jako słowniki będące elementami tychże list.
        :return:
        """
        self.network.append([0 for i in range(self.n_inputs)])

        for n_neurons in self.n_hidden:
            self.network.append([{'weights': [random.uniform(-1,1) for i in range(len(self.network[-1]) + 1)]}
                                 for neurons in range(n_neurons)])

        self.network.append([{'weights': [random.uniform(-1,1) for i in range(len(self.network[-1]) + 1)]}
                             for neurons in range(self.n_outputs)])
