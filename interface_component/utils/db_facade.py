import datetime

import pandas as pd
from pymongo import MongoClient, errors


class DBFacade:
    # Old
    # def __init__(self):
    #     self.db_name = 'test_db'
    #     self.client = MongoClient()
    #     self.db = getattr(self.client, self.db_name)
    #     self.collection_name = None
    #     self.raport_name = None
    #
    #     # inicjalizacja baz danych
    #     self.perceptron_sgd_collection = self.db.perceptron_sgd_collection
    #     self.perceptron_ga_collection = self.db.perceptron_ga_collection
    #     self.ann_bp_collection = self.db.ann_bp_collection
    #     self.ann_ga_collection = self.db.ann_ga_collection

    def __init__(self):
        self.db_name = 'test_db'
        self.client = self.db_connect()
        if self.client is not None:
            self.db = getattr(self.client, self.db_name)
            self.collection_name = None
            self.raport_name = None

            # inicjalizacja baz danych
            self.perceptron_sgd_collection = self.db.perceptron_sgd_collection
            self.perceptron_ga_collection = self.db.perceptron_ga_collection
            self.ann_bp_collection = self.db.ann_bp_collection
            self.ann_ga_collection = self.db.ann_ga_collection
        else:
            self.db = None
            self.collection_name = None
            self.raport_name = None

            # inicjalizacja baz danych
            self.perceptron_sgd_collection = None
            self.perceptron_ga_collection = None
            self.ann_bp_collection = None
            self.ann_ga_collection = None

    def db_connect(self):
        try:
            client = MongoClient(serverSelectionTimeoutMS=2000)
            client.server_info()
            return client
        except:
            return None

    # ==SERIALIZERY=====================================================================================================

    # def parse_gradient_simple_data(self, raport_name, train_metrics, test_metrics):
    #     raport = {}
    #     raport['date'] = datetime.datetime.now()
    #     raport['name'] = raport_name
    #     raport['test_metrics'] = test_metrics
    #     train_metrics['data'].index = train_metrics['data'].index.astype(str)
    #     data_dict = pd.DataFrame.to_dict(train_metrics['data'])
    #     train_metrics['data'] = data_dict
    #     raport['train_metrics'] = train_metrics
    #     return raport

    def cv_data_aggregator(self, train_metrics):
        new_data = []
        for metric in train_metrics:
            metric['data'].index = metric['data'].index.astype(str)
            metric_dict = pd.DataFrame.to_dict(metric['data'])
            metric['data'] = metric_dict
            new_data.append(metric)

        return new_data

    def simple_data_aggregator(self, train_metrics):
        train_metrics['data'].index = train_metrics['data'].index.astype(str)
        data_dict = pd.DataFrame.to_dict(train_metrics['data'])
        train_metrics['data'] = data_dict

        return train_metrics

    def raport_data_serializer(self, type, train_metrics, test_metrics):
        raport = {}
        raport['date'] = datetime.datetime.now()
        raport['name'] = type + " " + raport['date'].strftime("%Y-%m-%d %H:%M:%S")
        raport['test_metrics'] = test_metrics

        if isinstance(train_metrics, list):
            train_metrics = self.cv_data_aggregator(train_metrics)
        else:
            train_metrics = self.simple_data_aggregator(train_metrics)

        raport['train_metrics'] = train_metrics

        return raport

    # ==PARSERY=========================================================================================================
    def parse_list_data(self, train_metrics):
        all_data = []
        for metric in train_metrics:
            new_metrics = {}
            new_metrics['data'] = pd.DataFrame.from_dict(metric['data'])
            new_metrics['confusion_matrix'] = metric['confusion_matrix']
            all_data.append(new_metrics)

        return all_data

    def gradient_data_parser(self, document):
        test_metrics = document['test_metrics']
        # train_metrics = None

        if isinstance(document['train_metrics'], list):
            train_metrics = self.parse_list_data(document['train_metrics'])
        else:
            train_metrics = {}
            train_metrics['data'] = pd.DataFrame.from_dict(document['train_metrics']['data'])
            train_metrics['confusion_matrix'] = document['train_metrics']['confusion_matrix']

        return test_metrics, train_metrics

    def genetic_data_parser(self, document):
        train_metrics = pd.DataFrame.from_dict(document['train_metrics']['data'])

        return document['test_metrics'], train_metrics


    # ==ZAPISYWANIE=====================================================================================================

    # def save_gradient_simple_raport(self, raport_name, train_metrics, test_metrics):
    #     raport = self.parse_gradient_simple_data(raport_name, train_metrics, test_metrics)
    #     self.perceptron_simple_raports.insert_one(raport)
        # print()

    def save_raport(self, type, train_metrics, test_metrics):
        model_collection = {'perceptron_sgd':   self.perceptron_sgd_collection,
                            'perceptron_ga':    self.perceptron_ga_collection,
                            'ann_bp':           self.ann_bp_collection,
                            'ann_ga':           self.ann_ga_collection}

        if self.db is not None:
            raport = self.raport_data_serializer(type, train_metrics, test_metrics)

            model_collection[type].insert_one(raport)

            return True
        else:
            return False

    # ==WCZYTYWANIE=====================================================================================================
    def get_collections_list(self):
        # if self.db is not None:
        #     collection_list = self.db.list_collection_names()
        #     return collection_list
        # else:
        #     return ['Brak połączenia z bazą danych']
        try:
            collection_list = self.db.list_collection_names()
            return collection_list
        except:
            return ['Brak połączenia z bazą danych']


    def get_raport_list(self, type):
        model_collection = {'perceptron_sgd_collection':    self.perceptron_sgd_collection,
                            'perceptron_ga_collection':     self.perceptron_ga_collection,
                            'ann_bp_collection':            self.ann_bp_collection,
                            'ann_ga_collection':            self.ann_ga_collection}
        self.collection_name = type

        if self.db is not None:
            documents = model_collection[self.collection_name].find({})
            name_list = [document['name'] for document in documents]

            if len(name_list) == 0:
                name_list = ['Brak zapisanych raportów']

            return name_list
        else:
            return ['Brak połączenia z bazą danych']

    def get_raport_data(self, raport_name):
        model_collection = {'perceptron_sgd_collection':    self.perceptron_sgd_collection,
                            'perceptron_ga_collection':     self.perceptron_ga_collection,
                            'ann_bp_collection':            self.ann_bp_collection,
                            'ann_ga_collection':            self.ann_ga_collection}
        data_parser = {'perceptron_sgd_collection':     self.gradient_data_parser,
                       'ann_bp_collection':             self.gradient_data_parser,
                       'perceptron_ga_collection':      self.genetic_data_parser,
                       'ann_ga_collection':             self.genetic_data_parser}
        self.raport_name = raport_name

        documents = model_collection[self.collection_name].find({'name': raport_name})
        document = [document for document in documents][0]

        # test_metrics, train_metrics = self.gradient_data_parser(document)
        test_metrics, train_metrics = data_parser[self.collection_name](document)

        return test_metrics, train_metrics

    # ==USÓWANIE========================================================================================================
    def delete_raport(self):
        model_collection = {'perceptron_sgd_collection':    self.perceptron_sgd_collection,
                            'perceptron_ga_collection':     self.perceptron_ga_collection,
                            'ann_bp_collection':            self.ann_bp_collection,
                            'ann_ga_collection':            self.ann_ga_collection}

        if self.db is not None:
            model_collection[self.collection_name].delete_one({'name': self.raport_name})

            return self.raport_name
        else:
            return 'Brak połączenia z bazą danych'

# db_facade = DBFacade()
