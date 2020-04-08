import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, confusion_matrix,\
    roc_curve, auc, f1_score


class MetricPreprocessor:
    def __init__(self):
        pass

    # ==OBLICZANIE_METRYK===============================================================================================

    def calculate_mse(self, data):
        mse = []
        for epoch in data['n_epoch'].unique():
            epoch_data = data.loc[data['n_epoch'] == epoch]
            mean = mean_squared_error(epoch_data['real_value'], epoch_data['prediction'])

            mse.append(mean)

        return mse

    def calculate_mae(self, data):
        mae = []
        for epoch in data['n_epoch'].unique():
            epoch_data = data.loc[data['n_epoch'] == epoch]
            mean = mean_absolute_error(epoch_data['real_value'], epoch_data['prediction'])

            mae.append(mean)

        return mae

    def calculate_accuracy(self, data):
        accuracy = []

        for epoch in data['n_epoch'].unique():
            epoch_data = data.loc[data['n_epoch'] == epoch]
            mean = accuracy_score(epoch_data['real_value'], epoch_data['prediction'])

            accuracy.append(mean)

        return accuracy

    def calculate_confusion_matrix(self, data):
        tn, fp, fn, tp = confusion_matrix(data['real_value'], data['prediction']).ravel()
        # return [[tp, tn], [fp, fn]]
        return [[tp, fp], [tn, fn]]

    def calculate_auc(self, data):
        area = []
        for epoch in data['n_epoch'].unique():
            epoch_data = data.loc[data['n_epoch'] == epoch]

            fpr, tpr, treshold = roc_curve(epoch_data['real_value'], epoch_data['prediction'])
            epoch_area = auc(fpr, tpr)

            area.append(epoch_area)

        return area

    # ==PERCEPTRON_SGD==================================================================================================

    def perprocess_sgd_split_metrics(self, metrics, model_config):
        """
        Metoda odpowiada za obliczenie i agregację metryk dla zbioru treningowego i testowego, w procesie uczenia
        perceptronu algorytmem stochastycznego zejścia gradientem (sgd), przy zastosowaniu walidacji za pomocą zbioru
        testowego.
        :param metrics: dict
        :return: train_metrics: dict, test_metric: dict
        """
        train_metrics = {'data': pd.DataFrame()}
        test_metrics = {}

        train_data = metrics['data_train'][-1]
        test_data = metrics['data_test'][-1]
        train_metrics['data']['n_epoch'] = metrics['data_train'][-1]['n_epoch'].unique()

        # mse
        train_metrics['data']['mse'] = self.calculate_mse(train_data)
        test_metrics['mse'] = self.calculate_mse(test_data)[0]

        # mae
        train_metrics['data']['mae'] = self.calculate_mae(train_data)
        test_metrics['mae'] = self.calculate_mae(test_data)[0]

        # accuracy
        train_metrics['data']['accuracy'] = self.calculate_accuracy(train_data)
        test_metrics['accuracy'] = self.calculate_accuracy(test_data)[0]

        # confusion matrix
        train_metrics['confusion_matrix'] = self.calculate_confusion_matrix(train_data)
        test_metrics['confusion_matrix'] = self.calculate_confusion_matrix(test_data)

        return train_metrics, test_metrics

    def perprocess_sgd_cv_metrics(self, metrics, model_config):
        train_metrics = []
        test_metrics = []

        last_val_train_data = metrics['data_train'][-model_config['validation_mode']['k']:]
        last_val_test_data = metrics['data_test'][-model_config['validation_mode']['k']:]

        for i, (train_data, test_data) in enumerate(zip(last_val_train_data, last_val_test_data)):
            k_train = {'data': pd.DataFrame()}
            k_test = {}

            k_train['data']['n_epoch'] = train_data['n_epoch'].unique()

            # mse
            k_train['data']['mse'] = self.calculate_mse(train_data)
            k_test['mse'] = self.calculate_mse(test_data)[0]

            # mae
            k_train['data']['mae'] = self.calculate_mae(train_data)
            k_test['mae'] = self.calculate_mae(test_data)[0]

            # accuracy
            k_train['data']['accuracy'] = self.calculate_accuracy(train_data)
            k_test['accuracy'] = self.calculate_accuracy(test_data)[0]

            # confusion matrix
            k_train['confusion_matrix'] = self.calculate_confusion_matrix(train_data)
            k_test['confusion_matrix'] = self.calculate_confusion_matrix(test_data)

            train_metrics.append(k_train)
            test_metrics.append(k_test)

        return train_metrics, test_metrics

    def run_sgd(self, model_config):
        mode = {'simple_split':         self.perprocess_sgd_split_metrics,
                'cross_validation':     self.perprocess_sgd_cv_metrics}

        raport_data = mode[model_config['validation_mode']['mode']](model_config['metrics'], model_config)

        return raport_data

    # ==PERCEPTRON_GA===================================================================================================

    def perprocess_ga_metrics(self, model_config):
        # train_metrics = {'data': pd.DataFrame()}
        test_metrics = {'val_fit': sorted(model_config['metrics']['val_fit'])}

        train_metrics = model_config['metrics']['data_train'][-1]

        return train_metrics, test_metrics




    # def run_sgd(self, validation_mode, metrics, model_config):
    #     mode = {'simple_split':         self.perprocess_sgd_split_metrics,
    #             'cross_validation':     self.perceptron_sgd_cv_metrics}
    #
    #     raport_data = mode[validation_mode](metrics, model_config)
    #
    #     return raport_data

