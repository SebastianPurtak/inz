import copy

import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from interface_component.app import app
from interface_component.utils.db_facade import DBFacade
from interface_component.utils.raport_exporter import RaportExporter

layout = {}

train_metrics = pd.DataFrame()
test_metrics = {}


def set_metrics(train_data, test_data):
    global train_metrics
    global test_metrics

    train_metrics = train_data
    test_metrics = test_data


def generate_ann_bp_split_raport(back_link):
    global layout

    raport = dbc.Container([

        # ==NAGŁÓWEK====================================================================================================

        dbc.Row(id='header-ann-bp',
                children=[
                    dbc.Col([
                        html.Div([
                            html.H1('Śieć Neuronowa BP Raport')],
                            style={
                                'position': 'relative',
                                'top': '20%',
                                'textAlign': 'center'
                            })
                    ])
                ],
                style={
                    'height': '100px'}),

        dbc.Row(id='train_metrics-ann-bp-header',
                children=html.H4('Metryki procesu uczenia'),
                justify='center',
                style={'padding': '10px',
                       'margin-top': '30px'}),

        # ==ŚREDNIA_KWADRATÓW_BŁĘDÓW====================================================================================

        dbc.Row([
            dcc.Graph(id='mse-ann-bp-graph',
                      figure={'data': [{'x': train_metrics['data']['n_epoch'], 'y': train_metrics['data']['mse']}],
                              'layout': {'title': 'Mean Squared Error',
                                         'xaxis': {'title': 'Epoki'},
                                         'yaxis': {'title': 'MSE'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'}}})
        ], justify='center', style={'padding': '10px', }),

        # ==ŚREDINA_BŁĘDÓW_ABSOLUTNYCH======================================================================================

        dbc.Row([
            dcc.Graph(id='mea-ann-bp-graph',
                      figure={'data': [{'x': train_metrics['data']['n_epoch'], 'y': train_metrics['data']['mae']}],
                              'layout': {'title': 'Mean Absolute Error',
                                         'xaxis': {'title': 'Epoki'},
                                         'yaxis': {'title': 'MAE'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'}}})
        ], justify='center', style={'padding': '10px'}),

        # ==ACCURACY========================================================================================================

        dbc.Row([
            dcc.Graph(id='acc-ann-bp-graph',
                      figure={'data': [{'x': train_metrics['data']['n_epoch'], 'y': train_metrics['data']['accuracy']}],
                              'layout': {'title': 'Accuracy',
                                         'xaxis': {'title': 'Epoki'},
                                         'yaxis': {'title': 'ACC'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'}}})
        ], justify='center', style={'padding': '10px'}),

        # ==CONFUSION_MATRIX================================================================================================

        dbc.Row([
            dcc.Graph(id='cf-matrix-train-set-ann-bp',
                      figure={'data': [{'type': 'heatmap',
                                        'x': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'y': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'z': train_metrics['confusion_matrix'],
                                        'showscale': False}],
                              'layout': {'title': 'Confusion Matrix',
                                         'xaxis': {'title': 'Predykcje'},
                                         'yaxis': {'title': 'Prawdziwe wartości'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'},
                                         'annotations': [{'x': 'Klasa 1',
                                                          'y': 'Klasa 1',
                                                          'text': train_metrics['confusion_matrix'][0][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 1',
                                                          'text': train_metrics['confusion_matrix'][0][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 1',
                                                          'text': train_metrics['confusion_matrix'][0][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 2',
                                                          'text': train_metrics['confusion_matrix'][1][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 2',
                                                          'text': train_metrics['confusion_matrix'][1][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 2',
                                                          'text': train_metrics['confusion_matrix'][1][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 3',
                                                          'text': train_metrics['confusion_matrix'][2][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 3',
                                                          'text': train_metrics['confusion_matrix'][2][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 3',
                                                          'text': train_metrics['confusion_matrix'][2][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         ]}})
        ], justify='center', style={'padding': '10px'}),

        # ==METRYKI_ZBIORU_TESTOWEGO========================================================================================

        dbc.Row(id='test_metrics-ann-bp-header',
                children=html.H4('Metryki zbioru testowego'),
                justify='center',
                style={'padding': '10px',
                       'margin-bottom': '30px'}),

        dbc.Row(id='test_set_mse-ann-bp-row',
                children=[html.P('MSE: '), html.P(str(round(test_metrics['mse'], 5)), style={'text-indent': '50px'})],
                justify='center',
                ),

        dbc.Row(id='test_set_mae-ann-bp-row',
                children=[html.P('MAE: '), html.P(str(round(test_metrics['mae'], 5)), style={'text-indent': '50px'})],
                justify='center',
                ),

        dbc.Row(id='test_set_acc-ann-bp-row',
                children=[html.P('Accuracy: '),
                          html.P(str(round(test_metrics['accuracy'], 5)), style={'text-indent': '50px'})],
                justify='center',
                ),

        dbc.Row([
            dcc.Graph(id='cf-matrix-test-set-ann-bp',
                      figure={'data': [{'type': 'heatmap',
                                        'x': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'y': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'z': test_metrics['confusion_matrix'],
                                        'showscale': False}],
                              'layout': {'title': 'Confusion Matrix',
                                         'xaxis': {'title': 'Predykcje'},
                                         'yaxis': {'title': 'Prawdziwe wartości'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'},
                                         'annotations': [{'x': 'Klasa 1',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics['confusion_matrix'][0][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics['confusion_matrix'][0][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics['confusion_matrix'][0][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics['confusion_matrix'][1][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics['confusion_matrix'][1][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics['confusion_matrix'][1][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics['confusion_matrix'][2][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics['confusion_matrix'][2][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics['confusion_matrix'][2][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         ]}})
        ], justify='center', style={'padding': '10px'}),

        # ==STOPKA==========================================================================================================

        dbc.Row(dbc.Col(
            children=[dbc.Button(id='save-raport-ann-bp-button', children='Zapisz raport w bazie danych', color='secondary', size='lg',
                                 block=True)], width=4), justify='center', style={'padding': '10px'}),

        dbc.Row(
            dbc.Col(children=[dbc.Button(id='save-raport-ann-bp-json-button', children='Zapisz raport do pliku json',
                                         color='secondary', size='lg', block=True)], width=4), justify='center',
            style={'padding': '10px'}),

        dbc.Row(id='save-raport-ann-bp-alert', children=[], justify='center'),
        dbc.Row(id='save-raport-ann-bp-json-alert', children=[], justify='center'),

        dbc.Row(dbc.Col(children=[dbc.Button(id='back_to_config-ann-bp', children='Powrót', color='secondary',
                                             size='lg', block=True, href=back_link)], width=4), justify='center',
                style={'padding': '10px'}),

    ],
        fluid=True)

    layout = raport


@app.callback(Output('save-raport-ann-bp-alert', 'children'), [Input('save-raport-ann-bp-button', 'n_clicks')])
def save_ann_bp_raport(n_clicks):
    if n_clicks is not None:
        db_facade = DBFacade()

        if db_facade.save_raport('ann_bp', copy.deepcopy(train_metrics), copy.deepcopy(test_metrics)):
            return dbc.Alert(id='save-info', children='Zapisano raport', color='success')
        else:
            return dbc.Alert(id='save-info', children='Nie udało się zapisać raportu', color='danger')


@app.callback(Output('save-raport-ann-bp-json-alert', 'children'), [Input('save-raport-ann-bp-json-button', 'n_clicks')])
def save_ann_bp_json_raport(n_clicks):
    if n_clicks is not None:
        raport_exporter = RaportExporter()
        raport_exporter.to_json('ann_bp', copy.deepcopy(train_metrics), copy.deepcopy(test_metrics))

        return dbc.Alert(id='save-info', children='Raport zapisano w zapisane_raporty', color='success')

# ==CROSS_VALIDATION_RAPORT=============================================================================================

def get_folds_labels():
    return ['fold ' + str(idx) for idx in range(len(train_metrics))]


def generate_ann_bp_cv_raport(back_link):
    global layout

    folds_labels = get_folds_labels()

    metrics_labels = ['MSE', 'MAE', 'ACCURACY']

    raport = dbc.Container([

        # ==NAGŁÓWEK========================================================================================================

        dbc.Row(id='header-ann-bp',
                children=[
                    dbc.Col([
                        html.Div([
                            html.H1('Śieć Neuronowa BP Raport')],
                            style={
                                'position': 'relative',
                                'top': '20%',
                                'textAlign': 'center'
                            })
                    ])
                ],
                style={
                    'height': '100px',
                }),

        dbc.Row(id='train_metrics-ann-bp-header',
                children=html.H4('Metryki procesu uczenia'),
                justify='center',
                style={'padding': '10px',
                       'margin-top': '30px'}),

        # ==ŚREDNIA_KWADRATÓW_BŁĘDÓW========================================================================================

        dbc.Row([
            dcc.Graph(id='mse-cv-ann-bp-graph',
                      figure={
                          'data': [{'x': train_metrics[0]['data']['n_epoch'], 'y': train_metrics[0]['data']['mse']}],
                          'layout': {'title': 'Mean Squared Error',
                                     'xaxis': {'title': 'Epoki'},
                                     'yaxis': {'title': 'MSE'},
                                     'plot_bgcolor': 'rgba(0,0,0,0)',
                                     'paper_bgcolor': 'rgba(0,0,0,0)',
                                     'font': {'color': '#FFFFFF'}}})
        ], justify='center', style={'padding': '10px', }),

        dbc.Row(id='mse-ann-bp-fold-choice-label',
                children=[html.Label('Wybierz zbiór')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='mse-ann-bp-fold-select',
                         options=[{'label': value, 'value': value} for value in folds_labels],
                         value=folds_labels[0],
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

        # ==ŚREDINA_BŁĘDÓW_ABSOLUTNYCH======================================================================================

        dbc.Row([
            dcc.Graph(id='mae-cv-ann-bp-graph',
                      figure={
                          'data': [{'x': train_metrics[0]['data']['n_epoch'], 'y': train_metrics[0]['data']['mae']}],
                          'layout': {'title': 'Mean Absolute Error',
                                     'xaxis': {'title': 'Epoki'},
                                     'yaxis': {'title': 'MAE'},
                                     'plot_bgcolor': 'rgba(0,0,0,0)',
                                     'paper_bgcolor': 'rgba(0,0,0,0)',
                                     'font': {'color': '#FFFFFF'}}})
        ], justify='center', style={'padding': '10px'}),

        dbc.Row(id='mae-ann-bp-fold-choice-label',
                children=[html.Label('Wybierz zbiór')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='mae-ann-bp-fold-select',
                         options=[{'label': value, 'value': value} for value in folds_labels],
                         value=folds_labels[0],
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

        # ==ACCURACY========================================================================================================

        dbc.Row([
            dcc.Graph(id='acc-cv-ann-bp-graph',
                      figure={'data': [{'x': train_metrics[0]['data']['n_epoch'],
                                        'y': train_metrics[0]['data']['accuracy']}],
                              'layout': {'title': 'Accuracy',
                                         'xaxis': {'title': 'Epoki',
                                                   'tick0': 0,
                                                   'dtick': 1},
                                         'yaxis': {'title': 'ACC'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'}}})
        ], justify='center', style={'padding': '10px'}),

        dbc.Row(id='acc-ann-bp-fold-choice-label',
                children=[html.Label('Wybierz zbiór')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='acc-ann-bp-fold-select',
                         options=[{'label': value, 'value': value} for value in folds_labels],
                         value=folds_labels[0],
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

        # ==FOLDS_HISTOGRAM=================================================================================================

        dbc.Row([
            dcc.Graph(id='folds-ann-bp-histogram',
                      figure={
                          'data': [{'x': [i for i in range(len(folds_labels))],
                                    'y': [float(metrics['data']['mse'][-1:]) for metrics in train_metrics],
                                    'type': 'bar'}],
                          'layout': {'title': 'Histogram',
                                     'xaxis': {'title': 'K-folds'},
                                     'yaxis': {'title': 'MSE'},
                                     'plot_bgcolor': 'rgba(0,0,0,0)',
                                     'paper_bgcolor': 'rgba(0,0,0,0)',
                                     'font': {'color': '#FFFFFF'}}
                      })
        ], justify='center'),

        dbc.Row(id='histogram-choice-ann-bp-label',
                children=[html.Label('Wybierz metrykę')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='histogram-metrics-ann-bp-select',
                         options=[{'label': value, 'value': value} for value in metrics_labels],
                         value='MSE',
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

        # ==CONFUSION_MATRIX================================================================================================

        # dbc.Row([
        #     dcc.Graph(id='cf-matrix-ann-bp-cv-train-set',
        #               figure={'data': [{'type': 'heatmap',
        #                                 'x': ['True', 'False'],
        #                                 'y': ['Positive', 'Negative'],
        #                                 'z': train_metrics[0]['confusion_matrix'],
        #                                 'showscale': False}],
        #                       'layout': {'title': 'Confusion Matrix',
        #                                  'plot_bgcolor': 'rgba(0,0,0,0)',
        #                                  'paper_bgcolor': 'rgba(0,0,0,0)',
        #                                  'font': {'color': '#FFFFFF'},
        #                                  'annotations': [{'x': 'True',
        #                                                   'y': 'Positive',
        #                                                   'text': train_metrics[0]['confusion_matrix'][0][0],
        #                                                   'showarrow': False,
        #                                                   'font': {'color': 'white'}},
        #                                                  {'x': 'True',
        #                                                   'y': 'Negative',
        #                                                   'text': train_metrics[0]['confusion_matrix'][1][0],
        #                                                   'showarrow': False,
        #                                                   'font': {'color': 'white'}},
        #                                                  {'x': 'False',
        #                                                   'y': 'Positive',
        #                                                   'text': train_metrics[0]['confusion_matrix'][0][1],
        #                                                   'showarrow': False,
        #                                                   'font': {'color': 'white'}},
        #                                                  {'x': 'False',
        #                                                   'y': 'Negative',
        #                                                   'text': train_metrics[0]['confusion_matrix'][1][1],
        #                                                   'showarrow': False,
        #                                                   'font': {'color': 'white'}}]}})
        # ], justify='center', style={'padding': '10px'}),

        dbc.Row([
            dcc.Graph(id='cf-matrix-ann-bp-cv-train-set',
                      figure={'data': [{'type': 'heatmap',
                                        'x': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'y': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'z': train_metrics[0]['confusion_matrix'],
                                        'showscale': False}],
                              'layout': {'title': 'Confusion Matrix',
                                         'xaxis': {'title': 'Predykcje'},
                                         'yaxis': {'title': 'Prawdziwe wartości'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'},
                                         'annotations': [{'x': 'Klasa 1',
                                                          'y': 'Klasa 1',
                                                          'text': train_metrics[0]['confusion_matrix'][0][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 1',
                                                          'text': train_metrics[0]['confusion_matrix'][0][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 1',
                                                          'text': train_metrics[0]['confusion_matrix'][0][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 2',
                                                          'text': train_metrics[0]['confusion_matrix'][1][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 2',
                                                          'text': train_metrics[0]['confusion_matrix'][1][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 2',
                                                          'text': train_metrics[0]['confusion_matrix'][1][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 3',
                                                          'text': train_metrics[0]['confusion_matrix'][2][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 3',
                                                          'text': train_metrics[0]['confusion_matrix'][2][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 3',
                                                          'text': train_metrics[0]['confusion_matrix'][2][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         ]}})
        ], justify='center', style={'padding': '10px'}),

        dbc.Row(id='cf-cv-ann-bp-fold-choice-label',
                children=[html.Label('Wybierz zbiór')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='cf-cv-ann-bp-fold-select',
                         options=[{'label': value, 'value': value} for value in folds_labels],
                         value=folds_labels[0],
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

        # ==TEST_SET_HISTOGRAM==============================================================================================

        dbc.Row(id='test_cv-ann-bp-metrics_header',
                children=html.H4('Metryki zbioru testowego'),
                justify='center',
                style={'padding': '10px',
                       'margin-bottom': '30px',
                       'margin-top': '30px'}),

        dbc.Row([
            dcc.Graph(id='test-set-ann-bp-histogram',
                      figure={
                          'data': [{'x': [i for i in range(len(folds_labels))],
                                    'y': [float(metrics['mse']) for metrics in test_metrics],
                                    'type': 'bar'}],
                          'layout': {'title': 'Histogram',
                                     'xaxis': {'title': 'K-folds',
                                               'tick0': 0,
                                               'dtick': 1},
                                     'yaxis': {'title': 'MSE'},
                                     'plot_bgcolor': 'rgba(0,0,0,0)',
                                     'paper_bgcolor': 'rgba(0,0,0,0)',
                                     'font': {'color': '#FFFFFF'}}
                      })
        ], justify='center'),

        dbc.Row(id='ts-histogram-choice-ann-bp-label',
                children=[html.Label('Wybierz metrykę')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='ts-histogram-ann-bp-metrics-select',
                         options=[{'label': value, 'value': value} for value in metrics_labels],
                         value='MSE',
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

        # ==TS_CONFUSION_MATRIX=============================================================================================

        # dbc.Row([
        #     dcc.Graph(id='cf-matrix-cv-ann-bp-test-set',
        #               figure={'data': [{'type': 'heatmap',
        #                                 'x': ['True', 'False'],
        #                                 'y': ['Positive', 'Negative'],
        #                                 'z': test_metrics[0]['confusion_matrix'],
        #                                 'showscale': False}],
        #                       'layout': {'title': 'Confusion Matrix',
        #                                  'plot_bgcolor': 'rgba(0,0,0,0)',
        #                                  'paper_bgcolor': 'rgba(0,0,0,0)',
        #                                  'font': {'color': '#FFFFFF'},
        #                                  'annotations': [{'x': 'True',
        #                                                   'y': 'Positive',
        #                                                   'text': test_metrics[0]['confusion_matrix'][0][0],
        #                                                   'showarrow': False,
        #                                                   'font': {'color': 'white'}},
        #                                                  {'x': 'True',
        #                                                   'y': 'Negative',
        #                                                   'text': test_metrics[0]['confusion_matrix'][1][0],
        #                                                   'showarrow': False,
        #                                                   'font': {'color': 'white'}},
        #                                                  {'x': 'False',
        #                                                   'y': 'Positive',
        #                                                   'text': test_metrics[0]['confusion_matrix'][0][1],
        #                                                   'showarrow': False,
        #                                                   'font': {'color': 'white'}},
        #                                                  {'x': 'False',
        #                                                   'y': 'Negative',
        #                                                   'text': test_metrics[0]['confusion_matrix'][1][1],
        #                                                   'showarrow': False,
        #                                                   'font': {'color': 'white'}}]}})
        # ], justify='center', style={'padding': '10px'}),

        dbc.Row([
            dcc.Graph(id='cf-matrix-cv-ann-bp-test-set',
                      figure={'data': [{'type': 'heatmap',
                                        'x': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'y': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'z': test_metrics[0]['confusion_matrix'],
                                        'showscale': False}],
                              'layout': {'title': 'Confusion Matrix',
                                         'xaxis': {'title': 'Predykcje'},
                                         'yaxis': {'title': 'Prawdziwe wartości'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'},
                                         'annotations': [{'x': 'Klasa 1',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics[0]['confusion_matrix'][0][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics[0]['confusion_matrix'][0][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics[0]['confusion_matrix'][0][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics[0]['confusion_matrix'][1][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics[0]['confusion_matrix'][1][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics[0]['confusion_matrix'][1][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics[0]['confusion_matrix'][2][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics[0]['confusion_matrix'][2][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics[0]['confusion_matrix'][2][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         ]}})
        ], justify='center', style={'padding': '10px'}),

        dbc.Row(id='cf-cv-ts-ann-bp-fold-choice-label',
                children=[html.Label('Wybierz zbiór')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='cf-cv-ts-ann-bp-fold-select',
                         options=[{'label': value, 'value': value} for value in folds_labels],
                         value=folds_labels[0],
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center', style={'margin-bottom': '30px'}),

        # ==STOPKA==========================================================================================================

        dbc.Row(dbc.Col(
            children=[dbc.Button(id='save-raport-ann-bp-cv-button', children='Zapisz raport w bazie danych', color='secondary',
                                 size='lg', block=True)], width=4), justify='center', style={'padding': '10px'}),

        dbc.Row(
            dbc.Col(children=[dbc.Button(id='save-raport-ann-bp-cv-json-button', children='Zapisz raport do pliku json',
                                         color='secondary', size='lg', block=True)], width=4), justify='center',
            style={'padding': '10px'}),

        dbc.Row(id='save-raport-ann-bp-cv-alert', children=[], justify='center'),
        dbc.Row(id='save-raport-ann-bp-cv-json-alert', children=[], justify='center'),

        dbc.Row(dbc.Col(children=[dbc.Button(id='ann-bp-back_to_config', children='Powrót', color='secondary',
                                             size='lg', block=True, href=back_link)], width=4), justify='center',
                style={'padding': '10px'}),

    ],
        fluid=True)

    layout = raport


@app.callback(Output('mse-cv-ann-bp-graph', 'figure'), [Input('mse-ann-bp-fold-select', 'value')])
def update_ann_bp_mse_graph(value):
    folds_labels = get_folds_labels()
    fold_index = folds_labels.index(value)

    figure = {'data': [{'x': train_metrics[fold_index]['data']['n_epoch'], 'y': train_metrics[fold_index]['data']['mse']}],
                              'layout': {'title': 'Mean Squared Error',
                                         'xaxis': {'title': 'Epoki'},
                                         'yaxis': {'title': 'MSE'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'}}}

    return figure


@app.callback(Output('mae-cv-ann-bp-graph', 'figure'), [Input('mae-ann-bp-fold-select', 'value')])
def update_ann_bp_mae_graph(value):
    folds_labels = get_folds_labels()
    fold_index = folds_labels.index(value)

    figure = {'data': [{'x': train_metrics[fold_index]['data']['n_epoch'],
                        'y': train_metrics[fold_index]['data']['mae']}],
              'layout': {'title': 'Mean Absolute Error',
                         'xaxis': {'title': 'Epoki'},
                         'yaxis': {'title': 'MAE'},
                         'plot_bgcolor': 'rgba(0,0,0,0)',
                         'paper_bgcolor': 'rgba(0,0,0,0)',
                         'font': {'color': '#FFFFFF'}}}

    return figure


@app.callback(Output('acc-cv-ann-bp-graph', 'figure'), [Input('acc-ann-bp-fold-select', 'value')])
def update_ann_bp_acc_graph(value):
    folds_labels = get_folds_labels()
    fold_index = folds_labels.index(value)

    figure = {'data': [{'x': train_metrics[fold_index]['data']['n_epoch'],
                        'y': train_metrics[fold_index]['data']['accuracy']}],
              'layout': {'title': 'Accuracy',
                         'xaxis': {'title': 'Epoki',
                                   'tick0': 0,
                                   'dtick': 1},
                         'yaxis': {'title': 'ACC'},
                         'plot_bgcolor': 'rgba(0,0,0,0)',
                         'paper_bgcolor': 'rgba(0,0,0,0)',
                         'font': {'color': '#FFFFFF'}}}

    return figure


@app.callback(Output('cf-matrix-ann-bp-cv-train-set', 'figure'), [Input('cf-cv-ann-bp-fold-select', 'value')])
def update_cf_graph(value):
    folds_labels = get_folds_labels()
    fold_index = folds_labels.index(value)

    # figure = {'data': [{'type': 'heatmap',
    #                     'x': ['True', 'False'],
    #                     'y': ['Positive', 'Negative'],
    #                     'z': train_metrics[fold_index]['confusion_matrix'],
    #                     'showscale': False}],
    #           'layout': {'title': 'Confusion Matrix',
    #                      'plot_bgcolor': 'rgba(0,0,0,0)',
    #                      'paper_bgcolor': 'rgba(0,0,0,0)',
    #                      'font': {'color': '#FFFFFF'},
    #                      'annotations': [{'x': 'True',
    #                                       'y': 'Positive',
    #                                       'text': train_metrics[fold_index]['confusion_matrix'][0][0],
    #                                       'showarrow': False,
    #                                       'font': {'color': 'white'}},
    #                                      {'x': 'True',
    #                                       'y': 'Negative',
    #                                       'text': train_metrics[fold_index]['confusion_matrix'][1][0],
    #                                       'showarrow': False,
    #                                       'font': {'color': 'white'}},
    #                                      {'x': 'False',
    #                                       'y': 'Positive',
    #                                       'text': train_metrics[fold_index]['confusion_matrix'][0][1],
    #                                       'showarrow': False,
    #                                       'font': {'color': 'white'}},
    #                                      {'x': 'False',
    #                                       'y': 'Negative',
    #                                       'text': train_metrics[fold_index]['confusion_matrix'][1][1],
    #                                       'showarrow': False,
    #                                       'font': {'color': 'white'}}]}}

    figure = {'data': [{'type': 'heatmap',
                        'x': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                        'y': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                        'z': train_metrics[fold_index]['confusion_matrix'],
                        'showscale': False}],
              'layout': {'title': 'Confusion Matrix',
                         'xaxis': {'title': 'Predykcje'},
                         'yaxis': {'title': 'Prawdziwe wartości'},
                         'plot_bgcolor': 'rgba(0,0,0,0)',
                         'paper_bgcolor': 'rgba(0,0,0,0)',
                         'font': {'color': '#FFFFFF'},
                         'annotations': [{'x': 'Klasa 1',
                                          'y': 'Klasa 1',
                                          'text': train_metrics[fold_index]['confusion_matrix'][0][0],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 2',
                                          'y': 'Klasa 1',
                                          'text': train_metrics[fold_index]['confusion_matrix'][0][1],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 3',
                                          'y': 'Klasa 1',
                                          'text': train_metrics[fold_index]['confusion_matrix'][0][2],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 1',
                                          'y': 'Klasa 2',
                                          'text': train_metrics[fold_index]['confusion_matrix'][1][0],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 2',
                                          'y': 'Klasa 2',
                                          'text': train_metrics[fold_index]['confusion_matrix'][1][1],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 3',
                                          'y': 'Klasa 2',
                                          'text': train_metrics[fold_index]['confusion_matrix'][1][2],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 1',
                                          'y': 'Klasa 3',
                                          'text': train_metrics[fold_index]['confusion_matrix'][2][0],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 2',
                                          'y': 'Klasa 3',
                                          'text': train_metrics[fold_index]['confusion_matrix'][2][1],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 3',
                                          'y': 'Klasa 3',
                                          'text': train_metrics[fold_index]['confusion_matrix'][2][2],
                                          'showarrow': False,
                                          'font': {'color': 'white'}}]}}

    return figure


@app.callback(Output('folds-ann-bp-histogram', 'figure'), [Input('histogram-metrics-ann-bp-select', 'value')])
def update_histogram(value):
    folds_labels = get_folds_labels()

    print(value.lower())

    figure = {
        'data': [{'x': [i for i in range(len(folds_labels))],
                  'y': [float(metrics['data'][value.lower()][-1:]) for metrics in train_metrics],
                  'type': 'bar'}],
        'layout': {'title': 'Histogram',
                   'xaxis': {'title': 'K-folds',
                             'tick0': 0,
                             'dtick': 1},
                   'yaxis': {'title': value},
                   'plot_bgcolor': 'rgba(0,0,0,0)',
                   'paper_bgcolor': 'rgba(0,0,0,0)',
                   'font': {'color': '#FFFFFF'}}
    }

    return figure


@app.callback(Output('test-set-ann-bp-histogram', 'figure'), [Input('ts-histogram-ann-bp-metrics-select', 'value')])
def update_ts_histogram(value):
    folds_labels = get_folds_labels()

    print(value.lower())

    figure = {
        'data': [{'x': [i for i in range(len(folds_labels))],
                  'y': [float(metrics[value.lower()]) for metrics in test_metrics],
                  'type': 'bar'}],
        'layout': {'title': 'Histogram',
                   'xaxis': {'title': 'K-folds',
                             'tick0': 0,
                             'dtick': 1},
                   'yaxis': {'title': value},
                   'plot_bgcolor': 'rgba(0,0,0,0)',
                   'paper_bgcolor': 'rgba(0,0,0,0)',
                   'font': {'color': '#FFFFFF'}}
    }

    return figure


@app.callback(Output('cf-matrix-cv-ann-bp-test-set', 'figure'), [Input('cf-cv-ts-ann-bp-fold-select', 'value')])
def update_cf_cv_graph(value):
    folds_labels = get_folds_labels()
    fold_index = folds_labels.index(value)

    # figure = {'data': [{'type': 'heatmap',
    #                     'x': ['True', 'False'],
    #                     'y': ['Positive', 'Negative'],
    #                     'z': test_metrics[fold_index]['confusion_matrix'],
    #                     'showscale': False}],
    #           'layout': {'title': 'Confusion Matrix',
    #                      'plot_bgcolor': 'rgba(0,0,0,0)',
    #                      'paper_bgcolor': 'rgba(0,0,0,0)',
    #                      'font': {'color': '#FFFFFF'},
    #                      'annotations': [{'x': 'True',
    #                                       'y': 'Positive',
    #                                       'text': test_metrics[fold_index]['confusion_matrix'][0][0],
    #                                       'showarrow': False,
    #                                       'font': {'color': 'white'}},
    #                                      {'x': 'True',
    #                                       'y': 'Negative',
    #                                       'text': test_metrics[fold_index]['confusion_matrix'][1][0],
    #                                       'showarrow': False,
    #                                       'font': {'color': 'white'}},
    #                                      {'x': 'False',
    #                                       'y': 'Positive',
    #                                       'text': test_metrics[fold_index]['confusion_matrix'][0][1],
    #                                       'showarrow': False,
    #                                       'font': {'color': 'white'}},
    #                                      {'x': 'False',
    #                                       'y': 'Negative',
    #                                       'text': test_metrics[fold_index]['confusion_matrix'][1][1],
    #                                       'showarrow': False,
    #                                       'font': {'color': 'white'}}]}}

    figure = {'data': [{'type': 'heatmap',
                        'x': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                        'y': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                        'z': test_metrics[fold_index]['confusion_matrix'],
                        'showscale': False}],
              'layout': {'title': 'Confusion Matrix',
                         'xaxis': {'title': 'Predykcje'},
                         'yaxis': {'title': 'Prawdziwe wartości'},
                         'plot_bgcolor': 'rgba(0,0,0,0)',
                         'paper_bgcolor': 'rgba(0,0,0,0)',
                         'font': {'color': '#FFFFFF'},
                         'annotations': [{'x': 'Klasa 1',
                                          'y': 'Klasa 1',
                                          'text': test_metrics[fold_index]['confusion_matrix'][0][0],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 2',
                                          'y': 'Klasa 1',
                                          'text': test_metrics[fold_index]['confusion_matrix'][0][1],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 3',
                                          'y': 'Klasa 1',
                                          'text': test_metrics[fold_index]['confusion_matrix'][0][2],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 1',
                                          'y': 'Klasa 2',
                                          'text': test_metrics[fold_index]['confusion_matrix'][1][0],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 2',
                                          'y': 'Klasa 2',
                                          'text': test_metrics[fold_index]['confusion_matrix'][1][1],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 3',
                                          'y': 'Klasa 2',
                                          'text': test_metrics[fold_index]['confusion_matrix'][1][2],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 1',
                                          'y': 'Klasa 3',
                                          'text': test_metrics[fold_index]['confusion_matrix'][2][0],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 2',
                                          'y': 'Klasa 3',
                                          'text': test_metrics[fold_index]['confusion_matrix'][2][1],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'Klasa 3',
                                          'y': 'Klasa 3',
                                          'text': test_metrics[fold_index]['confusion_matrix'][2][2],
                                          'showarrow': False,
                                          'font': {'color': 'white'}}]}}

    return figure


@app.callback(Output('save-raport-ann-bp-cv-alert', 'children'), [Input('save-raport-ann-bp-cv-button', 'n_clicks')])
def save_ann_bp_cv_raport(n_clicks):
    if n_clicks is not None:
        db_facade = DBFacade()

        if db_facade.save_raport('ann_bp', copy.deepcopy(train_metrics), copy.deepcopy(test_metrics)):
            return dbc.Alert(id='save-info', children='Zapisano raport', color='success')
        else:
            return dbc.Alert(id='save-info', children='Nie udało się zapisać raportu', color='danger')


@app.callback(Output('save-raport-ann-bp-cv-json-alert', 'children'), [Input('save-raport-ann-bp-cv-json-button', 'n_clicks')])
def save_ann_bp_cv_json_raport(n_clicks):
    if n_clicks is not None:
        raport_exporter = RaportExporter()
        raport_exporter.to_json('ann_bp', copy.deepcopy(train_metrics), copy.deepcopy(test_metrics))

        return dbc.Alert(id='save-info', children='Raport zapisano w zapisane_raporty', color='success')