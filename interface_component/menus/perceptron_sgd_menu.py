import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from tqdm import tqdm

from interface_component.app import app
from interface_component.raports import perceptron_sgd_raport
from models_component.models_controller import ModelController
from utils.metrics_preprocessing import MetricPreprocessor


config = {'model':          'perceptron_sgd',
          'data_source':    'sonar_data',
          'model_config':   {'n_epoch':         0,
                             'l_rate':          0,
                             'validation_mode':  {'mode':           'simple_split',
                                                  'test_set_size':  0,
                                                  'k':              0},
                             'metrics':         {'data_train':      [],
                                                 'data_test':       [],
                                                 'cv_data_train':   [],
                                                 'cv_data_test':    [],
                                                 'n_epoch':         [],
                                                 'n_row':           [],
                                                 'prediction':      [],
                                                 'real_value':      [],
                                                 'error':           []}}}

metrisc_sgd = {}

clicks = 0

colors = {
    'background': '#D3D3D3',
}

data_sources = ['sonar_data', 'seed_data.csv']

validation_methods = ['cross_validation', 'simple_split']

# ==WERSJA_BOOTSRAP====================================================================================================

layout = dbc.Container([

    # ==NAGŁÓWEK=======================================================================================================

    dbc.Row(id='header',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Perceptron SGD')],
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

    # ==ŹRÓDŁO_DANYCH==================================================================================================

    dbc.Row(id='data_source-label',
            children=[html.Label('Wybierz źródło danych')],
            justify='center',
            style={'padding': '10px'}),

    dbc.Row(id='data_source-choice',
            children=[
                dcc.Dropdown(id='data_source-input',
                             options=[{'label': data_name, 'value': data_name} for data_name in data_sources],
                             clearable=False,
                             value=data_sources[0],
                             style={'width': '200px'})],
            justify='center'),

    # ==LICZBA_EPOK====================================================================================================

    dbc.Row(id='n_epoch-label',
            children=[html.Label(('Wybierz liczbę epok'))],
            justify="center",
            style={'padding': '10px'}),

    dbc.Row(id='n_epoch-choice',
            children=[
                dcc.Input(id='n_epoch-input',
                          value=10,
                          type='number',
                          min=0,
                          style={'width': '200px'})],
            justify="center"),

    # ==WSPÓŁCZYNNIK_UCZENIA===========================================================================================

    dbc.Row(id='l_rate-label',
            children=[html.Label(('Wybierz wartość współczynnika uczenia'))],
            justify="center",
            style={'padding': '10px'}),

    dbc.Row(id='l_rate-choice',
            children=[
                dcc.Input(id='l_rate-input',
                          value=0.01,
                          type='text',
                          min=0,
                          style={'width': '200px'})],
            justify='center'),

    # ==WALIDACJA======================================================================================================

    dbc.Row(id='validation-label-row',
            children=[html.Label('Wybierz metodę walidacji')],
            justify='center',
            style={'padding': '10px'}),

    dbc.Row(id='validation-choice-row', children=[
        dcc.Dropdown(id='validation-method-choice',
                     options=[
                         {'label': value, 'value': value} for value in validation_methods
                     ],
                     clearable=False,
                     value=validation_methods[1],
                     style={'width': '200px'})
    ],
            justify='center',
            style={'padding': '10px'}),

    # ==WALIDACJA_CONFIG===============================================================================================

    dbc.Row(id='validation-config-psgd-label',
            children=[],
            justify='center',
            style={'padding': '10px'}),

    dbc.Row(id='validation-config-psgd-row',
            children=[],
            justify='center',
            style={'padding': '10px'}),

    # ==PODGLĄD========================================================================================================

    # dbc.Row(html.Button(id='start-button-sgd', children='Start'), style={'padding': '10px'}, justify='center'),

    dbc.Row([dbc.Col([dbc.Button(id='start-button-sgd', children='Start', color='secondary', size='sm',
                                 block=True)], width=1)],
            justify='center', style={'padding': '15px'}),

    dbc.Row(html.Label('Komunikaty:'), justify='center'),

    dbc.Row(id='data_source-psgd-alert', children=[], justify='center'),
    dbc.Row(id='n_epoch-psgd-alert', children=[], justify='center'),
    dbc.Row(id='l_rate-psgd-alert', children=[], justify='center'),
    dbc.Row(id='val_method-psgd-error', children=[], justify='center'),
    dbc.Row(id='test_set-psgd-error', children=[], justify='center'),
    dbc.Row(id='k-psgd-error', children=[], justify='center'),
    dbc.Row(id='start-model-psgd-info', children=[], justify='center'),
    dbc.Row(id='end-model-psgd-info', children=[], justify='center'),

    dbc.Row(id='raport-button-psgd-row', children=[], justify='center', style={'padding': '15px'}),

    # dbc.Row([html.Button(id='back', children=[dcc.Link('Wróć', href='/')])], justify='center', style={'padding': '15px'})

    dbc.Row([dbc.Col([dbc.Button(id='back-button-sgd', children='Wróć', color='secondary', href='/models',
                                 size='sm', block=True)], width=1)],
            justify='center', style={'padding': '15px'}),
],
    fluid=True,
    style={
        'backgroundColor': '#D3D3D3'
    })

# ==VAL_CONFIG=========================================================================================================

k_label = html.Label('k zbiorów')

k_input = dcc.Input(id='k-input',
                    value=10,
                    type='number',
                    min=0,
                    style={'width': '200px'})

split_label = html.Label('Współczynnik podziału na zbiór treningowy i testowy')

split_input = dcc.Input(id='split-input',
                        value=0.25,
                        type='text',
                        min=0,
                        style={'width': '200px'})

# ==CALLBACKS==========================================================================================================


@app.callback(Output('data_source-psgd-alert', 'children'), [Input('data_source-input', 'value')])
def set_psgd_data_source(value):
    config['data_source'] = value


@app.callback(Output('n_epoch-psgd-alert', 'children'), [Input('n_epoch-input', 'value')])
def set_spgd_n_epoch(value):
    try:
        if value <= 0 or type(value) != int:
            return dbc.Alert(id='n_epoch-psgd_0_alert',
                             children='Liczba epok powinna być liczbą całkowitą większą od 0.',
                             color='danger')

        config['model_config']['n_epoch'] = value

    except:
        return dbc.Alert(id='n_epoch-psgd_None_alert', children='Podaj liczbę epok!', color='danger')


@app.callback(Output('l_rate-psgd-alert', 'children'), [Input('l_rate-input', 'value')])
def set_psgd_l_rate(value):
    try:
        if float(value) <= 0:
            return dbc.Alert(id='l_rate-psgd_0_alert',
                         children='Współczynnik uczenia powinien być liczbą większą od 0.',
                         color='danger')

        config['model_config']['l_rate'] = float(value)
        return ''
    except:
        return dbc.Alert(id='l_rate-psgd_None_alert',
                         children='Podaj wartość współczynnika uczenia.',
                         color='danger')


@app.callback([Output('validation-config-psgd-label', 'children'),
               Output('validation-config-psgd-row', 'children')],
              [Input('validation-method-choice', 'value')])
def set_val_method_psgd(value):
    val_config = {'cross_validation':   [k_label, k_input],
                  'simple_split':       [split_label, split_input]}

    config['model_config']['validation_mode']['mode'] = value

    return val_config[value][0], val_config[value][1]


@app.callback(Output('test_set-psgd-error', 'children'), [Input('split-input', 'value')])
def set_test_set_psgd(value):
    try:
        if float(value) >= 1 or float(value) <= 0:
            return dbc.Alert(id='test_set-psgd_Range_alert',
                         children='Współczynnik podziału na zbiór testowy i treningowy powinien być liczbą całkowitą z przedziału od 0 do 1.',
                         color='danger')
        else:
            config['model_config']['validation_mode']['test_set_size'] = float(value)

    except:
        return dbc.Alert(id='test_set-ann-bp_None_alert',
                         children='Podaj wartość współczynnika podziału na zbiór testowy i treningowy.',
                         color='danger')


@app.callback(Output('k-psgd-error', 'children'), [Input('k-input', 'value')])
def set_k_fold_psgd(value):
    try:
        if value <= 0 or type(value) != int:
            return dbc.Alert(id='k_fold-psgd_0_alert',
                             children='Wartość k powinna być liczbą większą od 0.',
                             color='danger')
        else:
            config['model_config']['validation_mode']['k'] = value
    except:
        return dbc.Alert(id='k_fold-psgd_None_alert', children='Podaj wartość k.',
                         color='danger')


@app.callback(Output('start-model-psgd-info', 'children'), [Input('start-button-sgd', 'n_clicks')])
def click_start_ann_bp_model(n_clicks):
    if n_clicks is not None:
        return dbc.Alert(id='progress-psgd-info', children='Trwa proces uczenia', color='primary')


@app.callback([Output('end-model-psgd-info', 'children'),
               Output('raport-button-psgd-row', 'children')],
              [Input('progress-psgd-info', 'children')])
def run_ann_bp_model(children):
    controller = ModelController()
    metrics_preprocessor = MetricPreprocessor()

    controller.run_model(config)

    raport_button = dbc.Row(
        [html.Button(id='raport-psgd', children=[dcc.Link('Pokaż raport', href='/models/perceptron_sgd_raport')])],
        justify='center',
        style={'padding': '15px'})

    train_metrics, test_metrics = metrics_preprocessor.run_sgd(config['model_config'])

    perceptron_sgd_raport.set_metrics(metrisc_sgd, train_metrics, test_metrics)

    if config['model_config']['validation_mode']['mode'] == 'simple_split':
        perceptron_sgd_raport.generate_raport()
    elif config['model_config']['validation_mode']['mode'] == 'cross_validation':
        perceptron_sgd_raport.generate_cv_raport()

    return dbc.Alert(id='result-psgd-info', children='Zakończono!', color='primary'), raport_button
