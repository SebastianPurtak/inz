import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from interface_component.app import app
from interface_component.raports import ann_bp_raport
from interface_component.utils.metrics_preprocessing import MetricPreprocessor
from interface_component.utils.data_management import DataManagement
from models_component.models_controller import ModelController

config = {'model':          'ann_bp',
          'data_source':    'seed_data',
          'model_config':   {'n_epoch':       100,
                            'l_rate':        0.3,
                            'n_hidden':      [5],
                           'validation_mode': {'mode':          'simple_split',
                                               'test_set_size':     0.25,
                                               'k':                 10},
                           'metrics': {'data_train':        [],
                                       'data_test':         [],
                                       'cv_data_train':     [],
                                       'cv_data_test':      [],
                                       'n_epoch':           [],
                                       'n_row':             [],
                                       'prediction':        [],
                                       'real_value':        [],
                                       'error':             [],
                                       'generation':        [],
                                       'best_fit':          [],
                                       'val_fit':           []}}}

data_manager = DataManagement()
data_sources = data_manager.get_datasets_list()

validation_methods = ['simple_split', 'cross_validation']

layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-ann-bp',
            children=[dbc.Col([html.Div([html.H1('Śieć Neuronowa BP')],
                                        style={'position': 'relative',
                                               'top': '20%',
                                               'textAlign': 'center',
                                               'margin-bottom': '75px'})])]),

    # ==ŹRUDŁO_DANYCH===================================================================================================

    dbc.Row(id='data_source-ann-bp-label',
            children=[html.Label('Wybierz źródło danych')],
            justify='center',
            style={'margin-top': '10px'}),

    dbc.Row(id='data_source-ann-bp-choice',
            children=[
                dcc.Dropdown(id='data_source-ann_bp-input',
                             options=[{'label': data_name, 'value': data_name} for data_name in data_sources],
                             clearable=False,
                             value=data_sources[0],
                             style={'width': '200px', 'color': '#000000'})],
            justify='center'),

    dbc.Row(dbc.Col([dbc.Button(id='refresh-button-ann_bp', children='Odświerz', color='secondary', size='sm',
                                block=True)], width=2), justify='center',
            style={'padding': '10px', 'margin-bottom': '35px'}),

    # ==LICZBA_EPOK=====================================================================================================

    dbc.Row(id='n_epoch-ann-bp-label',
            children=[html.Label(('Wybierz liczbę epok'))],
            justify="center"),

    dbc.Row(id='n_epoch-ann-bp-choice',
            children=[
                dcc.Input(id='n_epoch-ann-bp-input',
                          value=10,
                          type='number',
                          min=0,
                          style={'width': '200px'})],
            justify="center", style={'margin-bottom': '40px'}),

    # ==WSPÓŁCZYNNIK_UCZENIA===========================================================================================

    dbc.Row(id='l_rate-ann-bp-label',
            children=[html.Label(('Wybierz wartość współczynnika uczenia'))],
            justify="center"),

    dbc.Row(id='l_rate-ann-bp-choice',
            children=[
                dcc.Input(id='l_rate-ann-bp-input',
                          value=0.01,
                          type='text',
                          min=0,
                          style={'width': '200px'})],
            justify='center', style={'margin-bottom': '40px'}),

    # ==NEURONY_UKRYTE==================================================================================================

    dbc.Row(id='n_hidden-bp-label',
            children=[html.Label(('Wprowadź liczbę neuronów w warstwie ukrytej'))],
            justify="center"),

    dbc.Row(id='n_hidden-bp-choice',
            children=[
                dcc.Input(id='n_hidden-bp-input',
                          value=5,
                          type='number',
                          min=0,
                          style={'width': '200px'})],
            justify='center', style={'margin-bottom': '40px'}),

    # ==WALIDACJA=======================================================================================================

    dbc.Row(id='validation-ann-bp-label-row',
            children=[html.Label('Wybierz metodę walidacji')],
            justify='center'),

    dbc.Row(id='validation-choice-ann-bp-row', children=[
        dcc.Dropdown(id='validation-method-ann-bp-choice',
                     options=[
                         {'label': value, 'value': value} for value in validation_methods
                     ],
                     clearable=False,
                     value=validation_methods[0],
                     style={'width': '200px', 'color': '#000000'})
    ],
            justify='center',
            style={'margin-bottom': '40px'}),

    # ==WALIDACJA_CONFIG================================================================================================

    dbc.Row(id='validation-config-ann-bp-label',
            children=[],
            justify='center'),

    dbc.Row(id='validation-config-ann-bp-row',
            children=[],
            justify='center',
            style={'margin-bottom': '40px'}),

    # ==PODGLĄD=========================================================================================================

    dbc.Row([dbc.Col([dbc.Button(id='start-button-ann-bp', children='Start', color='secondary', size='sm',
                                 block=True)], width=2)],
            justify='center', style={'padding': '15px'}),

    dbc.Row(html.Label('Komunikaty:'), justify='center'),

    dbc.Row(id='data_source-ann-bp-alert', children=[], justify='center'),
    dbc.Row(id='data_refresh-ann-bp-alert', children=[], justify='center'),
    dbc.Row(id='n_epoch-ann-bp-alert', children=[], justify='center'),
    dbc.Row(id='l_rate-ann-bp-alert', children=[], justify='center'),
    dbc.Row(id='n_hidden-ann-bp-alert', children=[], justify='center'),
    dbc.Row(id='test_set-ann-bp-error', children=[], justify='center'),
    dbc.Row(id='k-ann-bp-error', children=[], justify='center'),
    dbc.Row(id='start-model-ann-bp-info', children=[], justify='center'),
    dbc.Row(id='end-model-ann-bp-info', children=[], justify='center'),

    dbc.Row(id='raport-button-ann-bp-row', children=[], justify='center', style={'padding': '15px'}),

    dbc.Row([dbc.Col([dbc.Button(id='back-ann-bp', children='Wróć', color='secondary', href='/models',
                                 size='sm', block=True)], width=2)],
            justify='center', style={'padding': '15px', 'margin-bottom': '20px'}),

],
    fluid=True)

# ==VAL_CONFIG=========================================================================================================

k_label = html.Label('k zbiorów')

k_input = dcc.Input(id='k-input-ann-bp',
                    value=10,
                    type='number',
                    min=0,
                    style={'width': '200px'})

split_label = html.Label('Współczynnik podziału na zbiór treningowy i testowy')

split_input = dcc.Input(id='split-input-ann-bp',
                        value=0.25,
                        type='text',
                        min=0,
                        style={'width': '200px'})

# ==CALLBACKS===========================================================================================================


@app.callback(Output('data_source-ann-bp-alert', 'children'), [Input('data_source-ann_bp-input', 'value')])
def set_data_source_ann_bp(value):
    config['data_source'] = value


@app.callback([Output('data_source-ann-bp-choice', 'children'), Output('data_refresh-ann-bp-alert', 'children')],
              [Input('refresh-button-ann_bp', 'n_clicks')])
def set_ann_bp_data_source(n_clicks):
    data_manager = DataManagement()
    global data_sources
    data_sources = data_manager.get_datasets_list()
    data_sources.remove('.gitkeep')

    drop_menu = dcc.Dropdown(id='data_source-ann_bp-input',
                             options=[{'label': data_name, 'value': data_name} for data_name in data_sources],
                             clearable=False,
                             value=data_sources[0],
                             style={'width': '200px', 'color': '#000000'})

    return drop_menu, None


@app.callback(Output('n_epoch-ann-bp-alert', 'children'), [Input('n_epoch-ann-bp-input', 'value')])
def set_n_epoch_ann_bp(value):
    try:
        if value <= 0 or type(value) != int:
            return dbc.Alert(id='n_epoch-ann-bp_0_alert',
                             children='Liczba epok powinna być liczbą całkowitą większą od 0.',
                             color='danger')

        config['model_config']['n_epoch'] = value
    except:
        return dbc.Alert(id='n_epoch-ann-bp_None_alert', children='Podaj liczbę epok!', color='danger')


@app.callback(Output('l_rate-ann-bp-alert', 'children'), [Input('l_rate-ann-bp-input', 'value')])
def set_l_rate_ann_bp(value):
    try:
        if float(value) <= 0:
            return dbc.Alert(id='l_rate-ann-bp_0_alert',
                         children='Współczynnik uczenia powinien być liczbą większą od 0.',
                         color='danger')
        config['model_config']['l_rate'] = float(value)
        return ''
    except:
        return dbc.Alert(id='l_rate-ann-bp_None_alert',
                         children='Podaj wartość współczynnika uczenia.',
                         color='danger')


@app.callback(Output('n_hidden-ann-bp-alert', 'children'), [Input('n_hidden-bp-input', 'value')])
def set_n_hidden_ann_bp(value):
    try:
        if value <= 0 or type(value) != int:
            return dbc.Alert(id='n_hidden-ann-bp_0_alert',
                             children='Liczba neuronów ukrytych powinna być liczbą całkowitą większą od 0.',
                             color='danger')

        config['model_config']['n_hidden'] = [value]
    except:
        return dbc.Alert(id='n_hidden-ann-bp_None_alert', children='Podaj liczbę neuronów w warstwie ukrytej',
                         color='danger')


@app.callback([Output('validation-config-ann-bp-label', 'children'),
               Output('validation-config-ann-bp-row', 'children')],
              [Input('validation-method-ann-bp-choice', 'value')])
def set_val_method_ann_bp(value):
    val_config = {'cross_validation':   [k_label, k_input],
                  'simple_split':       [split_label, split_input]}
    # print('test')
    config['model_config']['validation_mode']['mode'] = value

    return val_config[value][0], val_config[value][1]


@app.callback(Output('test_set-ann-bp-error', 'children'), [Input('split-input-ann-bp', 'value')])
def set_test_set_ann_bp(value):
    try:
        if float(value) >= 1 or float(value) <= 0:
            return dbc.Alert(id='test_set-ann-bp_Range_alert',
                     children='Współczynnik podziału na zbiór testowy i treningowy powinien być liczbą całkowitą z przedziału od 0 do 1.',
                     color='danger')
        else:
            config['model_config']['validation_mode']['test_set_size'] = float(value)
    except:
        return dbc.Alert(id='test_set-ann-bp_None_alert',
                         children='Podaj wartość współczynnika podziału na zbiór testowy i treningowy.',
                         color='danger')


@app.callback(Output('k-ann-bp-error', 'children'), [Input('k-input-ann-bp', 'value')])
def set_k_fold_ann_bp(value):
    try:
        if value <= 0 or type(value) != int:
            return dbc.Alert(id='k_fold-ann-bp_0_alert',
                             children='Wartość k powinna być liczbą większą od 0.',
                             color='danger')

        config['model_config']['validation_mode']['k'] = value
    except:
        return dbc.Alert(id='k_fold-ann-bp_None_alert', children='Podaj wartość k.',
                         color='danger')


@app.callback(Output('start-model-ann-bp-info', 'children'), [Input('start-button-ann-bp', 'n_clicks')])
def click_start_ann_bp_model(n_clicks):
    if n_clicks is not None:
        return dbc.Alert(id='progress-ann-bp-info', children='Trwa proces uczenia', color='primary')


@app.callback([Output('end-model-ann-bp-info', 'children'), Output('raport-button-ann-bp-row', 'children')],
              [Input('progress-ann-bp-info', 'children')])
def run_ann_bp_model(children):
    controller = ModelController()
    metrics_preprocessor = MetricPreprocessor()

    controller.run_model(config)

    raport_button = dbc.Col([dbc.Button(id='raport-ann_bp', children='Pokaż raport', color='secondary',
                                        href='/models/ann-bp_raport', size='sm', block=True)],
                            width=2)

    train_metrics, test_metrics = metrics_preprocessor.run_sgd(config['model_config'])

    ann_bp_raport.set_metrics(train_metrics, test_metrics)

    if config['model_config']['validation_mode']['mode'] == 'simple_split':
        ann_bp_raport.generate_ann_bp_split_raport('/models/ann_bp_menu')
    elif config['model_config']['validation_mode']['mode'] == 'cross_validation':
        ann_bp_raport.generate_ann_bp_cv_raport('/models/ann_bp_menu')

    return dbc.Alert(id='result-ann-bp-info', children='Zakończono!', color='primary'), raport_button








