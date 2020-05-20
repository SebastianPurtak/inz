import copy

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from interface_component.app import app
from interface_component.utils.db_facade import DBFacade
from interface_component.utils.raport_exporter import RaportExporter


layout = {}
train_metrics = None
test_metrics = None


def generate_raport(back_link, train_data, test_data):
    global layout
    global train_metrics
    global test_metrics

    train_metrics = train_data
    test_metrics = test_data

    raport = dbc.Container([

        # ==NAGŁÓWEK====================================================================================================

        dbc.Row(id='ann-ga-header',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Sieć Neuronowa GA Raport')],
                        style={
                            'position': 'relative',
                            'top': '20%',
                            'textAlign': 'center'
                        })
                ])
            ],
            style={
                'height': '100px'}),

        dbc.Row(id='train-metrics-ann-ga-header',
                children=html.H4('Metryki procesu uczenia'),
                justify='center',
                style={'padding': '10px',
                       'margin-top': '30px'}),

        # ==FITNESS=====================================================================================================

        dbc.Row(
            dcc.Graph(id='best_fit-ann-ga',
                      figure={'data': [{'x': train_metrics['generation'], 'y': train_metrics['best_fit']}],
                              'layout': {'title': 'Best Fitness',
                                         'xaxis': {'title': 'Pokolenia'},
                                         'yaxis': {'title': 'Best Fitness'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'}}}),
        justify='center', style={'padding': '10px'}),

        dbc.Row(
            dcc.Graph(id='mean_fit-ann-ga',
                      figure={'data': [{'x': train_metrics['generation'], 'y': train_metrics['mean_fit']}],
                              'layout': {'title': 'Mean Fitness',
                                         'xaxis': {'title': 'Pokolenia'},
                                         'yaxis': {'title': 'Mean Fitness'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'}}}),
            justify='center', style={'padding': '10px'}),

        dbc.Row([
            dcc.Graph(id='cf-matrix-ann-bp-cv-train-set',
                      figure={'data': [{'type': 'heatmap',
                                        'x': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'y': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'z': test_metrics['train_cv'],
                                        'showscale': False}],
                              'layout': {'title': 'Confusion Matrix',
                                         'xaxis': {'title': 'Predykcje'},
                                         'yaxis': {'title': 'Prawdziwe wartości'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'},
                                         'annotations': [{'x': 'Klasa 1',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics['train_cv'][0][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics['train_cv'][0][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics['train_cv'][0][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics['train_cv'][1][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics['train_cv'][1][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics['train_cv'][1][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics['train_cv'][2][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics['train_cv'][2][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics['train_cv'][2][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         ]}})
        ], justify='center', style={'padding': '10px'}),

        # ==METRYKI_ZBIORU_TESTOWEGO====================================================================================

        dbc.Row(id='test_metrics-ann-ga-header',
                children=html.H4('Metryki zbioru testowego'),
                justify='center',
                style={'padding': '10px',
                       'margin-bottom': '30px'}),

        dbc.Row(id='test_pop_best_fit-ann_ga-row- table', children=[
            html.Table([
                html.Thead(html.Tr(html.Th('Najlepsze wyniki populacji testowej'))),
                html.Tbody([
                    html.Tr(html.Td(str(result), style={'text-align': 'center'})) for result in test_metrics['val_fit']
                ])
            ])
        ], justify='center', style={'margin-bottom': '30px'}),

        dbc.Row([
            dcc.Graph(id='cf-matrix-ann-bp-cv-train-set',
                      figure={'data': [{'type': 'heatmap',
                                        'x': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'y': ['Klasa 1', 'Klasa 2', 'Klasa 3'],
                                        'z': test_metrics['test_cv'],
                                        'showscale': False}],
                              'layout': {'title': 'Confusion Matrix',
                                         'xaxis': {'title': 'Predykcje'},
                                         'yaxis': {'title': 'Prawdziwe wartości'},
                                         'plot_bgcolor': 'rgba(0,0,0,0)',
                                         'paper_bgcolor': 'rgba(0,0,0,0)',
                                         'font': {'color': '#FFFFFF'},
                                         'annotations': [{'x': 'Klasa 1',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics['test_cv'][0][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics['test_cv'][0][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 1',
                                                          'text': test_metrics['test_cv'][0][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics['test_cv'][1][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics['test_cv'][1][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 2',
                                                          'text': test_metrics['test_cv'][1][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 1',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics['test_cv'][2][0],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 2',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics['test_cv'][2][1],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         {'x': 'Klasa 3',
                                                          'y': 'Klasa 3',
                                                          'text': test_metrics['test_cv'][2][2],
                                                          'showarrow': False,
                                                          'font': {'color': 'white'}},
                                                         ]}})
        ], justify='center', style={'padding': '10px'}),

        # ==STOPKA======================================================================================================

        dbc.Row(dbc.Col(
            children=[dbc.Button(id='save-raport-ann_ga-button', children='Zapisz raport w bazie danych', color='secondary', size='lg',
                                 block=True)], width=4), justify='center', style={'padding': '10px'}),

        dbc.Row(
            dbc.Col(children=[dbc.Button(id='save-raport-ann-ga-json-button', children='Zapisz raport do pliku json',
                                         color='secondary', size='lg', block=True)], width=4), justify='center',
            style={'padding': '10px'}),

        dbc.Row(id='save-raport-ann_ga-alert', children=[], justify='center'),
        dbc.Row(id='save-raport-ann_ga-json-alert', children=[], justify='center'),

        # dbc.Row([html.Button(id='back_to_config-ann-ga', children=[dcc.Link('Pokaż config', href=back_link)])],
        #         justify='center',
        #         style={'padding': '15px'}),

        dbc.Row(dbc.Col(children=[dbc.Button(id='back_to_config-ann-ga', children='Powrót', color='secondary', size='lg',
                                             block=True, href=back_link)], width=4), justify='center',
                style={'padding': '10px'}),
    ],
    fluid=True)

    layout = raport


@app.callback(Output('save-raport-ann_ga-alert', 'children'), [Input('save-raport-ann_ga-button', 'n_clicks')])
def save_ann_ga_raport(n_clicks):
    if n_clicks is not None:
        db_facade = DBFacade()

        if db_facade.save_raport('ann_ga', {'data': train_metrics}, copy.deepcopy(test_metrics)):
            return dbc.Alert(id='save-info', children='Zapisano raport', color='success')
        else:
            return dbc.Alert(id='save-info', children='Nie udało się zapisać raportu', color='danger')


@app.callback(Output('save-raport-ann_ga-json-alert', 'children'), [Input('save-raport-ann-ga-json-button', 'n_clicks')])
def save_ann_ga_json_raport(n_clicks):
    if n_clicks is not None:
        raport_exporter = RaportExporter()
        raport_exporter.to_json('ann_ga', {'data': train_metrics}, copy.deepcopy(test_metrics))

        return dbc.Alert(id='save-info', children='Raport zapisano w zapisane_raporty', color='success')