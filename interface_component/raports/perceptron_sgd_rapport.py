import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from interface_component.app import app

metrics_sgd = {'mse': []}
train_metrics = pd.DataFrame()
test_metrics = {}

layout = {}

# layout = dbc.Container([
#     # ==NAGŁÓWEK=======================================================================================================
#
#     dbc.Row(id='header',
#             children=[
#                 dbc.Col([
#                     html.Div([
#                         html.H1('Perceptron SGD Raport')],
#                         style={
#                             'position': 'relative',
#                             'top': '20%',
#                             'textAlign': 'center'
#                         })
#                 ])
#             ],
#             style={
#                 'height': '100px',
#                 'backgroundColor': '#C0C0C0',
#             }),
#
#     # ==PIERWSZY_WYKRES=============================================================================================
#
#     dbc.Row([
#         dcc.Graph(id='first_graph',
#                   figure={'data': [{'x': [i for i in range(10)], 'y': metrics_sgd['mse'], 'type': 'bar'}],
#                           'layout': {'title': 'First graph'}})
#     ]),
#
#     dbc.Row(id='first_graph'),
#
#     # ==TESTY=======================================================================================================
#
#     dbc.Row([html.Button(id='test', children='Pokaż config')],
#                             justify='center',
#                             style={'padding': '15px'}),
#
#     dbc.Row(id='config_test',
#             children=[
#                 html.Div(id='config-test-div')
#             ])
# ],
#     fluid=True,
#     style={'backgroundColor': '#D3D3D3'})


@app.callback(Output('config-test-div', 'children'), [Input('test', 'n_clicks')])
def click_start_button(n_clicks):
    # metric = metrisc_sgd

    # metrisc_sgd = perceptron_menu.metrisc_sgd

    metrics = html.P(str(metrics_sgd))

    print('w raporcie: ', metrics_sgd)

    return metrics


@app.callback(Output('first_graph', 'children'), [Input('test', 'n_clicks')])
def generate_graph(n_clicks):
    return dcc.Graph(id='first_graph',
                     figure={'data': [{'x': [i for i in range(10)], 'y': metrics_sgd['mse']}],
                             'layout': {'title': 'First graph'}})


def set_metrics(metrics_data, train_data, test_data):
    global metrics_sgd
    global train_metrics
    global test_metrics

    train_metrics = train_data
    test_metrics = test_data

    metrics_sgd = metrics_data


def generate_raport():
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
                'backgroundColor': '#C0C0C0',
            }),

    # ==PIERWSZY_WYKRES================================================================================================

    dbc.Row([
        dcc.Graph(id='mse-graph',
                  figure={'data': [{'x': train_metrics['data']['n_epoch'], 'y': train_metrics['data']['mse']}],
                          'layout': {'title': 'Mean Squared Error',
                                     'xaxis': {'title': 'Epoki',
                                               },
                                     'yaxis': {'title': 'MSE'},
                                     'plot_bgcolor': '#D3D3D3',
                                     'paper_bgcolor': '#D3D3D3'}})
    ], justify='center', style={'padding': '10px', }),

    # ==ŚREDINA_BŁĘDÓW_ABSOLUTNYCH======================================================================================

    dbc.Row([
        dcc.Graph(id='mea-graph',
                  figure={'data': [{'x': train_metrics['data']['n_epoch'], 'y': train_metrics['data']['mae']}],
                          'layout': {'title': 'Mean Absolute Error',
                                     'xaxis': {'title': 'Epoki',
                                               'tick0': 0,
                                               'dtick': 1},
                                     'yaxis': {'title': 'MAE'},
                                     'plot_bgcolor': '#D3D3D3',
                                     'paper_bgcolor': '#D3D3D3'}})
    ], justify='center', style={'padding': '10px'}),

    # ==ACCURACY========================================================================================================

    dbc.Row([
        dcc.Graph(id='acc-graph',
                  figure={'data': [{'x': train_metrics['data']['n_epoch'], 'y': train_metrics['data']['accuracy']}],
                          'layout': {'title': 'Accuracy',
                                     'xaxis': {'title': 'Epoki',
                                               'tick0': 0,
                                               'dtick': 1},
                                     'yaxis': {'title': 'ACC'},
                                     'plot_bgcolor': '#D3D3D3',
                                     'paper_bgcolor': '#D3D3D3'}})
        ], justify='center', style={'padding': '10px'}),

    # ==CONFUSION_MATRIX================================================================================================

    dbc.Row([
        dcc.Graph(id='cf-matrix',
                  figure={'data': [{'type': 'heatmap',
                                    'x': ['True', 'False'],
                                    'y': ['Positive', 'Negative'],
                                    'z': train_metrics['confusion_matrix'],
                                    'showscale': False}],
                          'layout': {'title': 'Confusion Matrix',
                                     'plot_bgcolor': '#D3D3D3',
                                     'paper_bgcolor': '#D3D3D3',
                                     'annotations':[{'x': 'True',
                                                     'y': 'Positive',
                                                     # 'text': train_metrics['confusion_matrix'][0][0],
                                                     'text': train_metrics['confusion_matrix'][0][0],
                                                     'showarrow': False,
                                                     'font': {'color': 'white'}},
                                                    {'x': 'True',
                                                     'y': 'Negative',
                                                     # 'text': train_metrics['confusion_matrix'][0][1],
                                                     'text': train_metrics['confusion_matrix'][1][0],
                                                     'showarrow': False,
                                                     'font': {'color': 'white'}},
                                                    {'x': 'False',
                                                     'y': 'Positive',
                                                     # 'text': train_metrics['confusion_matrix'][1][0],
                                                     'text': train_metrics['confusion_matrix'][0][1],
                                                     'showarrow': False,
                                                     'font': {'color': 'white'}},
                                                    {'x': 'False',
                                                     'y': 'Negative',
                                                     # 'text': train_metrics['confusion_matrix'][1][1],
                                                     'text': train_metrics['confusion_matrix'][1][1],
                                                     'showarrow': False,
                                                     'font': {'color': 'white'}}]}})
        ], justify='center', style={'padding': '10px'}),

    # ==TESTY=======================================================================================================

    dbc.Row([html.Button(id='back_to_config', children=[dcc.Link('Pokaż config', href='perceptron_sgd_menu')])],
                            justify='center',
                            style={'padding': '15px'}),

    dbc.Row(id='config_test',
            children=[
                html.Div(id='config-test-div')
            ])
],
    fluid=True,
    style={'backgroundColor': '#D3D3D3'})

    layout = raport


# dbc.Row([
#         html.Button(id='back', children=[dcc.Link('Wróć', href='/')])
#     ]

