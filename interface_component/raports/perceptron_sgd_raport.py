import copy

import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from interface_component.app import app
from interface_component.utils.db_facade import DBFacade
from interface_component.utils.raport_exporter import RaportExporter

train_metrics = pd.DataFrame()
test_metrics = {}
raport_name = None

layout = {}


def set_metrics(train_data, test_data):
    global train_metrics
    global test_metrics

    train_metrics = train_data
    test_metrics = test_data


def generate_raport(back_link):
    global layout

    raport = dbc.Container([
    # ==NAGŁÓWEK=======================================================================================================

    dbc.Row(id='header',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Perceptron SGD Raport')],
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

    dbc.Row(id='train_metrics_header',
            children=html.H4('Metryki procesu uczenia'),
            justify='center',
            style={'padding': '10px',
                   'margin-top': '30px'}),

    # ==ŚREDNIA_KWADRATÓW_BŁĘDÓW========================================================================================

    dbc.Row([
        dcc.Graph(id='mse-graph',
                  figure={'data': [{'x': train_metrics['data']['n_epoch'], 'y': train_metrics['data']['mse']}],
                          'layout': {'title': 'Mean Squared Error',
                                     'xaxis': {'title': 'Epoki'},
                                     'yaxis': {'title': 'MSE'},
                                     'plot_bgcolor': 'rgba(0,0,0,0)',
                                     'paper_bgcolor': 'rgba(0,0,0,0)',
                                     'font': {'color': '#FFFFFF'}
                                     }})
    ], justify='center', style={'padding': '10px', }),

    # ==ŚREDINA_BŁĘDÓW_ABSOLUTNYCH======================================================================================

    dbc.Row([
        dcc.Graph(id='mea-graph',
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
        dcc.Graph(id='acc-graph',
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
        dcc.Graph(id='cf-matrix-train-set',
                  figure={'data': [{'type': 'heatmap',
                                    'x': ['True', 'False'],
                                    'y': ['Positive', 'Negative'],
                                    'z': train_metrics['confusion_matrix'],
                                    'showscale': False}],
                          'layout': {'title': 'Confusion Matrix',
                                     'plot_bgcolor': 'rgba(0,0,0,0)',
                                     'paper_bgcolor': 'rgba(0,0,0,0)',
                                     'font': {'color': '#FFFFFF'},
                                     'annotations':[{'x': 'True',
                                                     'y': 'Positive',
                                                     'text': train_metrics['confusion_matrix'][0][0],
                                                     'showarrow': False,
                                                     'font': {'color': 'white'}},
                                                    {'x': 'True',
                                                     'y': 'Negative',
                                                     'text': train_metrics['confusion_matrix'][1][0],
                                                     'showarrow': False,
                                                     'font': {'color': 'white'}},
                                                    {'x': 'False',
                                                     'y': 'Positive',
                                                     'text': train_metrics['confusion_matrix'][0][1],
                                                     'showarrow': False,
                                                     'font': {'color': 'white'}},
                                                    {'x': 'False',
                                                     'y': 'Negative',
                                                     'text': train_metrics['confusion_matrix'][1][1],
                                                     'showarrow': False,
                                                     'font': {'color': 'white'}}]}})
        ], justify='center', style={'padding': '10px'}),

    # ==METRYKI_ZBIORU_TESTOWEGO========================================================================================

    dbc.Row(id='test_metrics_header',
            children=html.H4('Metryki zbioru testowego'),
            justify='center',
            style={'padding': '10px',
                   'margin-bottom': '30px'}),

    dbc.Row(id='test_set_mse_row',
            children=[html.P('MSE: '), html.P(str(round(test_metrics['mse'], 5)), style={'text-indent': '50px'})],
            justify='center',
            ),

    dbc.Row(id='test_set_mae_row',
            children=[html.P('MAE: '), html.P(str(round(test_metrics['mae'], 5)), style={'text-indent': '50px'})],
            justify='center',
            ),

    dbc.Row(id='test_set_acc_row',
            children=[html.P('Accuracy: '), html.P(str(round(test_metrics['accuracy'], 5)), style={'text-indent': '50px'})],
            justify='center',
            ),

    dbc.Row([
        dcc.Graph(id='cf-matrix-test-set',
                  figure={'data': [{'type': 'heatmap',
                                    'x': ['True', 'False'],
                                    'y': ['Positive', 'Negative'],
                                    'z': test_metrics['confusion_matrix'],
                                    'showscale': False}],
                          'layout': {'title': 'Confusion Matrix',
                                     'plot_bgcolor': 'rgba(0,0,0,0)',
                                     'paper_bgcolor': 'rgba(0,0,0,0)',
                                     'font': {'color': '#FFFFFF'},
                                     'annotations': [{'x': 'True',
                                                      'y': 'Positive',
                                                      'text': test_metrics['confusion_matrix'][0][0],
                                                      'showarrow': False,
                                                      'font': {'color': 'white'}},
                                                     {'x': 'True',
                                                      'y': 'Negative',
                                                      'text': test_metrics['confusion_matrix'][1][0],
                                                      'showarrow': False,
                                                      'font': {'color': 'white'}},
                                                     {'x': 'False',
                                                      'y': 'Positive',
                                                      'text': test_metrics['confusion_matrix'][0][1],
                                                      'showarrow': False,
                                                      'font': {'color': 'white'}},
                                                     {'x': 'False',
                                                      'y': 'Negative',
                                                      'text': test_metrics['confusion_matrix'][1][1],
                                                      'showarrow': False,
                                                      'font': {'color': 'white'}}]}})
        ], justify='center', style={'padding': '10px'}),

    # ==STOPKA==========================================================================================================

    dbc.Row(dbc.Col(children=[dbc.Button(id='save-raport-psgd-button', children='Zapisz raport w bazie danych',
                                         color='secondary', size='lg', block=True)], width=4), justify='center',
            style={'padding': '10px'}),

    dbc.Row(dbc.Col(children=[dbc.Button(id='save-raport-psgd-csv-button', children='Zapisz raport do pliku json',
                                         color='secondary', size='lg', block=True)], width=4), justify='center',
            style={'padding': '10px'}),


    dbc.Row(id='save-raport-psgd-alert', children=[], justify='center'),
    dbc.Row(id='save-raport-psgd-csv-alert', children=[], justify='center'),

    dbc.Row(dbc.Col(children=[dbc.Button(id='back_to_config', children='Powrót', color='secondary', size='lg',
                                         block=True, href=back_link)], width=4), justify='center',
                style={'padding': '10px'}),

    dbc.Row(id='config_test',
            children=[
                html.Div(id='config-test-div')
            ])
],
    fluid=True)

    layout = raport


@app.callback(Output('save-raport-psgd-alert', 'children'), [Input('save-raport-psgd-button', 'n_clicks')])
def save_psgd_raport(n_clicks):
    if n_clicks is not None:
        db_facade = DBFacade()
        if db_facade.save_raport('perceptron_sgd', copy.deepcopy(train_metrics), copy.deepcopy(test_metrics)):
            return dbc.Alert(id='save-info', children='Zapisano raport', color='success')
        else:
            return dbc.Alert(id='save-info', children='Nie udało się zapisać raportu', color='danger')


@app.callback(Output('save-raport-psgd-csv-alert', 'children'), [Input('save-raport-psgd-csv-button', 'n_clicks')])
def save_psgd_json_raport(n_clicks):
    if n_clicks is not None:
        raport_exporter = RaportExporter()
        raport_exporter.to_json('perceptron_sgd', copy.deepcopy(train_metrics), copy.deepcopy(test_metrics))

        return dbc.Alert(id='save-info', children='Raport zapisano w zapisane_raporty', color='success')



# ==CROSS_VALIDATION_RAPORT=============================================================================================

def get_folds_labels():
    return ['fold ' + str(idx) for idx in range(len(train_metrics))]

def generate_cv_raport(back_link):
    global layout

    folds_labels = get_folds_labels()

    metrics_labels = ['MSE', 'MAE', 'ACCURACY']

    raport = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

        dbc.Row(id='header',
                children=[
                    dbc.Col([
                        html.Div([
                            html.H1('Perceptron SGD Raport')],
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

        dbc.Row(id='train_metrics_header',
                children=html.H4('Metryki procesu uczenia'),
                justify='center',
                style={'padding': '10px',
                       'margin-top': '30px'}),

    # ==ŚREDNIA_KWADRATÓW_BŁĘDÓW========================================================================================

        dbc.Row([
            dcc.Graph(id='mse-cv-graph',
                      figure={'data': [{'x': train_metrics[0]['data']['n_epoch'], 'y': train_metrics[0]['data']['mse']}],
                              'layout': {'title': 'Mean Squared Error',
                                         'xaxis': {'title': 'Epoki'},
                                         'yaxis': {'title': 'MSE'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'}}})
        ], justify='center', style={'padding': '10px', }),

        dbc.Row(id='mse-fold-choice-label',
                children=[html.Label('Wybierz zbiór')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='mse-fold-select',
                         options=[{'label': value, 'value': value} for value in folds_labels],
                         value=folds_labels[0],
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

    # ==ŚREDINA_BŁĘDÓW_ABSOLUTNYCH======================================================================================

        dbc.Row([
            dcc.Graph(id='mae-cv-graph',
                      figure={'data': [{'x': train_metrics[0]['data']['n_epoch'], 'y': train_metrics[0]['data']['mae']}],
                              'layout': {'title': 'Mean Absolute Error',
                                         'xaxis': {'title': 'Epoki'},
                                         'yaxis': {'title': 'MAE'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'}}})
        ], justify='center', style={'padding': '10px'}),

        dbc.Row(id='mae-fold-choice-label',
                children=[html.Label('Wybierz zbiór')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='mae-fold-select',
                         options=[{'label': value, 'value': value} for value in folds_labels],
                         value=folds_labels[0],
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

    # ==ACCURACY========================================================================================================

        dbc.Row([
            dcc.Graph(id='acc-cv-graph',
                      figure={'data': [{'x': train_metrics[0]['data']['n_epoch'],
                                        'y': train_metrics[0]['data']['accuracy']}],
                              'layout': {'title': 'Accuracy',
                                         'xaxis': {'title': 'Epoki'},
                                         'yaxis': {'title': 'ACC'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'}}})
        ], justify='center', style={'padding': '10px'}),

        dbc.Row(id='acc-fold-choice-label',
                children=[html.Label('Wybierz zbiór')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='acc-fold-select',
                         options=[{'label': value, 'value': value} for value in folds_labels],
                         value=folds_labels[0],
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

    # ==FOLDS_HISTOGRAM=================================================================================================

        dbc.Row([
           dcc.Graph(id='folds-histogram',
                     figure={
                         'data': [{'x': [i for i in range(len(folds_labels))],
                                   'y': [float(metrics['data']['mse'][-1:]) for metrics in train_metrics],
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

        dbc.Row(id='histogram-choice-label',
                children=[html.Label('Wybierz metrykę')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='histogram-metrics-select',
                         options=[{'label': value, 'value': value} for value in metrics_labels],
                         value='MSE',
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

    # ==CONFUSION_MATRIX================================================================================================

        dbc.Row([
            dcc.Graph(id='cf-matrix-cv-train-set',
                      figure={'data': [{'type': 'heatmap',
                                        'x': ['True', 'False'],
                                        'y': ['Positive', 'Negative'],
                                        'z': train_metrics[0]['confusion_matrix'],
                                        'showscale': False}],
                              'layout': {'title': 'Confusion Matrix',
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'},
                                         'annotations': [{'x': 'True',
                                                          'y': 'Positive',
                                                          'text': train_metrics[0]['confusion_matrix'][0][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'True',
                                                          'y': 'Negative',
                                                          'text': train_metrics[0]['confusion_matrix'][1][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'False',
                                                          'y': 'Positive',
                                                          'text': train_metrics[0]['confusion_matrix'][0][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'False',
                                                          'y': 'Negative',
                                                          'text': train_metrics[0]['confusion_matrix'][1][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}}]}})
        ], justify='center', style={'padding': '10px'}),

        dbc.Row(id='cf-cv-fold-choice-label',
                children=[html.Label('Wybierz zbiór')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='cf-cv-fold-select',
                         options=[{'label': value, 'value': value} for value in folds_labels],
                         value=folds_labels[0],
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

    # ==TEST_SET_HISTOGRAM==============================================================================================

        dbc.Row(id='test_cv_metrics_header',
                children=html.H4('Metryki zbioru testowego'),
                justify='center',
                style={'padding': '10px',
                       'margin-bottom': '30px',
                       'margin-top': '30px'}),

        dbc.Row([
            dcc.Graph(id='test-set-histogram',
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

        dbc.Row(id='ts-histogram-choice-label',
                children=[html.Label('Wybierz metrykę')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='ts-histogram-metrics-select',
                         options=[{'label': value, 'value': value} for value in metrics_labels],
                         value='MSE',
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center'),

    # ==TS_CONFUSION_MATRIX=============================================================================================

        dbc.Row([
            dcc.Graph(id='cf-matrix-cv-test-set',
                      figure={'data': [{'type': 'heatmap',
                                        'x': ['True', 'False'],
                                        'y': ['Positive', 'Negative'],
                                        'z': test_metrics[0]['confusion_matrix'],
                                        'showscale': False}],
                              'layout': {'title': 'Confusion Matrix',
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'},
                                         'annotations': [{'x': 'True',
                                                          'y': 'Positive',
                                                          'text': test_metrics[0]['confusion_matrix'][0][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'True',
                                                          'y': 'Negative',
                                                          'text': test_metrics[0]['confusion_matrix'][1][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'False',
                                                          'y': 'Positive',
                                                          'text': test_metrics[0]['confusion_matrix'][0][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'False',
                                                          'y': 'Negative',
                                                          'text': test_metrics[0]['confusion_matrix'][1][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}}]}})
        ], justify='center', style={'padding': '10px'}),

        dbc.Row(id='cf-cv-ts-fold-choice-label',
                children=[html.Label('Wybierz zbiór')],
                justify='center'),

        dbc.Row([
            dcc.Dropdown(id='cf-cv-ts-fold-select',
                         options=[{'label': value, 'value': value} for value in folds_labels],
                         value=folds_labels[0],
                         clearable=False,
                         style={'width': '200px', 'color': '#000000'})
        ], justify='center', style={'margin-bottom': '40px'}),

    # ==STOPKA==========================================================================================================

        dbc.Row(dbc.Col(
            children=[dbc.Button(id='save-raport-cv-button', children='Zapisz raport w bazie danych', color='secondary', size='lg',
                                 block=True)], width=4), justify='center', style={'padding': '10px'}),

        dbc.Row(dbc.Col(children=[dbc.Button(id='save-raport-psgd-cv-json-button', children='Zapisz raport do pliku json',
                                             color='secondary', size='lg', block=True)], width=4), justify='center',
                style={'padding': '10px'}),

        dbc.Row(id='save-raport-cv-alert', children=[], justify='center'),
        dbc.Row(id='save-raport-cv-json-alert', children=[], justify='center'),

        dbc.Row(dbc.Col(children=[dbc.Button(id='back_to_config', children='Powrót', color='secondary', size='lg',
                                             block=True, href=back_link)], width=4), justify='center',
                style={'padding': '10px'}),

    ],
        fluid=True)

    layout = raport


@app.callback(Output('mse-cv-graph', 'figure'), [Input('mse-fold-select', 'value')])
def update_mse_graph(value):
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


@app.callback(Output('mae-cv-graph', 'figure'), [Input('mae-fold-select', 'value')])
def update_mae_graph(value):
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


@app.callback(Output('acc-cv-graph', 'figure'), [Input('acc-fold-select', 'value')])
def update_acc_graph(value):
    folds_labels = get_folds_labels()
    fold_index = folds_labels.index(value)

    figure = {'data': [{'x': train_metrics[fold_index]['data']['n_epoch'],
                        'y': train_metrics[fold_index]['data']['accuracy']}],
              'layout': {'title': 'Accuracy',
                         'xaxis': {'title': 'Epoki'},
                         'yaxis': {'title': 'ACC'},
                         'plot_bgcolor': 'rgba(0,0,0,0)',
                         'paper_bgcolor': 'rgba(0,0,0,0)',
                         'font': {'color': '#FFFFFF'}}}

    return figure


@app.callback(Output('cf-matrix-cv-train-set', 'figure'), [Input('cf-cv-fold-select', 'value')])
def update_cf_graph(value):
    folds_labels = get_folds_labels()
    fold_index = folds_labels.index(value)

    figure = {'data': [{'type': 'heatmap',
                        'x': ['True', 'False'],
                        'y': ['Positive', 'Negative'],
                        'z': train_metrics[fold_index]['confusion_matrix'],
                        'showscale': False}],
              'layout': {'title': 'Confusion Matrix',
                         'plot_bgcolor': 'rgba(0,0,0,0)',
                         'paper_bgcolor': 'rgba(0,0,0,0)',
                         'font': {'color': '#FFFFFF'},
                         'annotations': [{'x': 'True',
                                          'y': 'Positive',
                                          'text': train_metrics[fold_index]['confusion_matrix'][0][0],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'True',
                                          'y': 'Negative',
                                          'text': train_metrics[fold_index]['confusion_matrix'][1][0],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'False',
                                          'y': 'Positive',
                                          'text': train_metrics[fold_index]['confusion_matrix'][0][1],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'False',
                                          'y': 'Negative',
                                          'text': train_metrics[fold_index]['confusion_matrix'][1][1],
                                          'showarrow': False,
                                          'font': {'color': 'white'}}]}}

    return figure


@app.callback(Output('folds-histogram', 'figure'), [Input('histogram-metrics-select', 'value')])
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


@app.callback(Output('test-set-histogram', 'figure'), [Input('ts-histogram-metrics-select', 'value')])
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


@app.callback(Output('cf-matrix-cv-test-set', 'figure'), [Input('cf-cv-ts-fold-select', 'value')])
def update_cf_cv_graph(value):
    folds_labels = get_folds_labels()
    fold_index = folds_labels.index(value)

    figure = {'data': [{'type': 'heatmap',
                        'x': ['True', 'False'],
                        'y': ['Positive', 'Negative'],
                        'z': test_metrics[fold_index]['confusion_matrix'],
                        'showscale': False}],
              'layout': {'title': 'Confusion Matrix',
                         'plot_bgcolor': 'rgba(0,0,0,0)',
                         'paper_bgcolor': 'rgba(0,0,0,0)',
                         'font': {'color': '#FFFFFF'},
                         'annotations': [{'x': 'True',
                                          'y': 'Positive',
                                          'text': test_metrics[fold_index]['confusion_matrix'][0][0],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'True',
                                          'y': 'Negative',
                                          'text': test_metrics[fold_index]['confusion_matrix'][1][0],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'False',
                                          'y': 'Positive',
                                          'text': test_metrics[fold_index]['confusion_matrix'][0][1],
                                          'showarrow': False,
                                          'font': {'color': 'white'}},
                                         {'x': 'False',
                                          'y': 'Negative',
                                          'text': test_metrics[fold_index]['confusion_matrix'][1][1],
                                          'showarrow': False,
                                          'font': {'color': 'white'}}]}}

    return figure


@app.callback(Output('save-raport-cv-alert', 'children'), [Input('save-raport-cv-button', 'n_clicks')])
def save_psgd_cv_raport(n_clicks):
    if n_clicks is not None:
        db_facade = DBFacade()

        if db_facade.save_raport('perceptron_sgd', copy.deepcopy(train_metrics), copy.deepcopy(test_metrics)):
            return dbc.Alert(id='save-info', children='Zapisano raport', color='success')
        else:
            return dbc.Alert(id='save-info', children='Nie udało się zapisać raportu', color='danger')


@app.callback(Output('save-raport-cv-json-alert', 'children'), [Input('save-raport-psgd-cv-json-button', 'n_clicks')])
def save_psgdcv_json_raport(n_clicks):
    if n_clicks is not None:
        raport_exporter = RaportExporter()
        raport_exporter.to_json('perceptron_sgd', copy.deepcopy(train_metrics), copy.deepcopy(test_metrics))

        return dbc.Alert(id='save-info', children='Raport zapisano w zapisane_raporty', color='success')

