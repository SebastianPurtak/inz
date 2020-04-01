import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, confusion_matrix,\
    roc_curve, auc, f1_score


class MetricPreprocessor:
    def __init__(self):
        pass

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

    def preprocess_split_sgd_metrics(self, metrics):
        train_metrics = {'data': pd.DataFrame()}
        # train_metrics_data = pd.DataFrame()
        test_metrics = {}

        train_data = metrics['data_train'][-1]
        test_data = metrics['data_test'][-1]

        # train_metrics_data['n_epoch'] = metrics['data_train'][0]['n_epoch'].unique()
        train_metrics['data']['n_epoch'] = metrics['data_train'][-1]['n_epoch'].unique()

        # mse
        # train_metrics_data['mse'] = self.calculate_mse(train_data)
        train_metrics['data']['mse'] = self.calculate_mse(train_data)
        test_metrics['mse'] = self.calculate_mse(test_data)[0]

        # mae
        # train_metrics_data['mae'] = self.calculate_mae(train_data)
        train_metrics['data']['mae'] = self.calculate_mae(train_data)
        test_metrics['mae'] = self.calculate_mae(test_data)[0]

        # accuracy
        # train_metrics_data['accuracy'] = self.calculate_accuracy(train_data)
        train_metrics['data']['accuracy'] = self.calculate_accuracy(train_data)
        test_metrics['accuracy'] = self.calculate_accuracy(test_data)[0]

        # confusion matrix
        train_metrics['confusion_matrix'] = self.calculate_confusion_matrix(train_data)
        test_metrics['confusion_matrix'] = self.calculate_confusion_matrix(test_data)

        # auc - narazie nie stosujemy, można się zapoznać z tą koncepcją
        # train_metrics['data']['auc'] = self.calculate_auc(train_data)
        # test_metrics['auc'] = self.calculate_auc(test_data)

        return train_metrics, test_metrics

    def run_sgd(self, validation_mode, metrics):
        mode = {'simple_split': self.preprocess_split_sgd_metrics}

        raport_data = mode[validation_mode](metrics)

        return raport_data

        # train_metrics = pd.DataFrame()
        # test_metrics = pd.DataFrame()
        #
        # train_metrics['n_epoch'] = metrics['data_train'][0]['n_epoch'].unique()
        # test_metrics['n_epoch'] = metrics['data_train'][0]['n_epoch'].unique()
        #
        # # mse
        # train_metrics['mse'] = self.calculate_mse(train_metrics)
        #
        #
        # sgd_metrics = {}
        # sgd_metrics = metrics['data_train'][0]
        #
        # m = pd.DataFrame()
        # m['n_epoch'] = metrics['data_train'][0]['n_epoch'].unique()
        #
        # # metrics['data_train'][0]['n_epoch'].unique()
        #
        # train_metrics = metrics['data_train'][0]
        #
        # mse = self.calculate_mse(train_metrics)
        #
        # sgd_metrics['mse'] = mse
        # return sgd_metrics
