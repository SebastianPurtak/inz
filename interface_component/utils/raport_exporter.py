import json
import base64
import os


from interface_component.utils.db_facade import DBFacade


class RaportExporter(DBFacade):

    def __init__(self):
        pass

    def to_json(self, type, train_metrics, test_metrics):
        raport = self.raport_data_serializer(type, train_metrics, test_metrics)
        filepath = os.path.join(os.getcwd() + '/zapisane_raporty/')
        filename = raport['name']
        raport['date'] = raport['date'].strftime("%Y-%m-%d %H:%M:%S")
        filepath = os.path.join(filepath + filename)

        with open(filepath, 'w') as file:
            json.dump(raport, file)

    def from_json(self, data):
        parser = {'perceptron_sgd':     self.gradient_data_parser,
                  'ann_bp':             self.gradient_data_parser,
                  'perceptron_ga':      self.genetic_data_parser,
                  'ann_ga':             self.genetic_data_parser}

        data_type, data_string = data.split(',')

        data_decoded = base64.b64decode(data_string)

        raport = json.loads(data_decoded)

        raport_type = raport['name'].split(' ')[0]

        test_metrics, train_metrics = parser[raport_type](raport)

        return raport_type, test_metrics, train_metrics