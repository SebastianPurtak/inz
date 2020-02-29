import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

from models_component.models.ANN.ann import NeuralNetwork


class NeuralNetworkBP:

    def __init__(self):
        pass

    # test - co z weryfikacją

    # =================================================================================================================
    # GROMADZENIE METRYK
    # =================================================================================================================

    def clear_metrics(self, metrics):
        metrics['n_epoch'] = []
        metrics['n_row'] = []
        metrics['prediction'] = []
        metrics['real_value'] = []

    # metody gromadzenia metryk
    def collect_metrics(self, n_epoch, n_row, prediction, real_value, metrics):
        metrics['n_epoch'].append(n_epoch)
        metrics['n_row'].append(n_row)
        metrics['prediction'].append(prediction)
        metrics['real_value'].append(real_value)

    def aggregate_metrics(self, model_config, data_target):
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

        # outputs.index((max(outputs)))
        # predictions = [pred.index((max(pred))) for pred in model_config['metrics']['prediction']]
        # real_value = [real.index((max(real))) for real in model_config['metrics']['real_value']]
        pass

    # =================================================================================================================
    # WALIDACJA MODELU
    # =================================================================================================================

    def test(self, test_set, model, model_config):
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
        for i in range(len(model)):
            if i != 0:
                inputs = [neuron['output'] for neuron in model[i - 1]]
            for neuron in model[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
        pass

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    # wsteczna propagacja błędu
    def error_propagate(self, model, answer):
        for i in reversed(range(len(model))):
            layer = model[i]
            errors = []

            # propagacja dla warstw ukrytych
            if i != len(model) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in model[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            # propagacja dla pierwszej warstwy
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(answer[j] - neuron['output'])

            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    # train
    def train(self, train_set, model, model_config):
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
        self.test(test_set, model, model_config)

    # walidacja krzyżowa
    def cross_validation(self, _, data, model_config):
        kf = KFold(n_splits=model_config['validation_mode']['k'])

        for i, (train_index, test_index) in enumerate(kf.split(data)):
            # perceptron = Perceptron(len(data.iloc[0][:-1]))
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
        n_inputs = len(data.loc[0]) - 1
        n_hidden = model_config['n_hidden']
        n_outputs = len(data.iloc[:, -1].unique())

        model = NeuralNetwork(n_inputs, n_hidden, n_outputs)
        model.create_network()

        return model

    def run(self, data, model_config):
        print("Start backpropagation")
        validation_method ={'simple_split':         self.simple_split,
                            'cross_validation':     self.cross_validation}

        model = self.init_model(data, model_config)

        validation_method[model_config['validation_mode']['mode']](model, data, model_config)


        pass