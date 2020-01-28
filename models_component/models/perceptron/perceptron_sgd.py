import random


class PerceptronSGDCore:

    def __init__(self, n_inputs):
        self.weights = [random.uniform(0,1) for input in range(n_inputs + 1)]

    def rectified_linear_activation_function(self, activation):
        if activation >= 0:
            return 1
        else:
            return 0

    def predict(self, data_row):
        activation = 0

        for i, input in enumerate(data_row):
            activation += self.weights[i + 1] * input

        activation += self.weights[0]

        return self.rectified_linear_activation_function(activation)


class PerceptronSGDUtils:

    def __init__(self):
        print()

    # Funkcje do dalszego przetworzenia danych

    # Funkcje do treningu
    def train(self, data, n_epoch, l_rate, perceptron):
        for epoch in range(n_epoch):
            error_sum = 0

            for idx, row in data.iterrows():
                prediction = perceptron.predict(row[:-1])

                error = row[-1:] - prediction
                error_sum += error**2

                perceptron.weights[0] += l_rate * error

                for i, x in enumerate(row[:-1]):
                    perceptron.weights[i + 1] += l_rate * error * x
            print('epoch: ', epoch, 'error: ', error_sum)

    # Ewentualna funkcja do testowania
    def test(self, data, perceptron):
        for idx, row in data.iterrows():
            prediction = perceptron.predict(row)
            print('prediction:', prediction, 'expected: ', row[-1:])

    def run(self, data):
        print('Start perceptron SGD')
        perceptron = PerceptronSGDCore(len(data.iloc[0][:-1]))

        data = data.sample(frac=1)
        train = data.iloc[:150, :]
        test = data.iloc[150:,:]

        self.train(train, 50, 0.01, perceptron)
        self.test(test, perceptron)


        # data = self.model_controller.get_data()

        print('data: ', data)