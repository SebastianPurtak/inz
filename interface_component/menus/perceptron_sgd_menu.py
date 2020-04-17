import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from tqdm import tqdm

from interface_component.app import app
from interface_component.raports import perceptron_sgd_raport
from models_component.models_controller import ModelController
from utils.metrics_preprocessing import MetricPreprocessor

# TODO: sprawdzić czy poniższe zmienne są faktycznie potrzebne
clicks_counter = 0
test = False

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

    dbc.Row(id='validation-config-label',
            children=[],
            justify='center',
            style={'padding': '10px'}),

    dbc.Row(id='validation-config-row',
            children=[],
            justify='center',
            style={'padding': '10px'}),

    # ==PODGLĄD========================================================================================================

    dbc.Row(html.Button(id='start-button-sgd', children='Start'), style={'padding': '10px'}, justify='center'),

    dbc.Row(html.Label('Komunikaty:'), justify='center'),

    dbc.Row(html.Label(id='data_source-error', children=''), justify='center', style={'backgroundColor': '#C0C0C0'}),
    dbc.Row(html.Label(id='n_epoch-error', children=''), justify='center', style={'backgroundColor': '#C0C0C0'}),
    dbc.Row(html.Label(id='l_rate-error', children=''), justify='center', style={'backgroundColor': '#C0C0C0'}),
    dbc.Row(html.Label(id='val_method-error', children=''), justify='center', style={'backgroundColor': '#C0C0C0'}),
    dbc.Row(html.Label(id='k-error', children=''), justify='center', style={'backgroundColor': '#C0C0C0'}),
    dbc.Row(html.Label(id='test_set-error', children=''), justify='center', style={'backgroundColor': '#C0C0C0'}),
    dbc.Row(html.Label(id='model-progress', children=''), justify='center', style={'backgroundColor': '#C0C0C0'}),
    dbc.Row(html.Label(id='model-end', children=''), justify='center', style={'backgroundColor': '#C0C0C0'}),
    dbc.Row(html.Label(id='model-start', children=''), justify='center', style={'backgroundColor': '#C0C0C0'}),

    # dbc.Row(html.Label(id='model-start1', children=''), justify='center', style={'backgroundColor': '#C0C0C0'}),
    # dbc.Row(dbc.Progress(id='progress'), justify='center', style={'backgroundColor': '#C0C0C0'}),

    dbc.Row([
        html.Button(id='back', children=[dcc.Link('Wróć', href='/')])
    ],
    justify='center',
    style={
        'padding': '15px'
    })
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


@app.callback(Output('data_source-error', 'children'), [Input('data_source-input', 'value')])
def set_data_source(value):
    # Metoda walidacji wyboru źródła danych, może okazać się niepotrzebna
    try:
        if value is None:
            raise
        config['data_source'] = value
        return ''
    except:
        return 'Należy wybrać źródło danych.'


@app.callback(Output('n_epoch-error', 'children'), [Input('n_epoch-input', 'value')])
def set_data_n_epoch(value):
    try:
        n_epoch = value

        if type(n_epoch) != int:
            raise

        config['model_config']['n_epoch'] = n_epoch

        return ''
    except:
        return 'Wprowadzono nieprawidłową liczbę epok. Wprowadź liczbę całkowitą.'


@app.callback(Output('l_rate-error', 'children'), [Input('l_rate-input', 'value')])
def set_l_rate(value):
    try:
        config['model_config']['l_rate'] = float(value)
        return ''
    except:
        return 'Wprowadzono nieprawidłową wartość wspólczynnika uczenia. Wprowadź liczbę zmiennoprzecinkową.'


@app.callback(Output('val_method-error', 'children'), [Input('validation-method-choice', 'value')])
def set_val_method(value):
    try:
        val_method = value

        if val_method == None:
            raise

        config['model_config']['validation_mode']['mode'] = val_method
        return ''
    except:
        return 'Nie wybrano żadnej metody walidacji.'

@app.callback([Output('validation-config-label', 'children'),
               Output('validation-config-row', 'children')],
              [Input('validation-method-choice', 'value')])
def update_val_method_config(value):
    val_config = {'cross_validation':       [k_label, k_input],
                  'simple_split':    [split_label, split_input]}
    try:
        if value == None:
            raise

        return val_config[value][0], val_config[value][1]
    except:
        return '', ''


@app.callback(Output('test_set-error', 'children'), [Input('split-input', 'value')])
def set_test_set_size(value):
    try:
        test_set_size = float(value)

        if test_set_size > 1 or test_set_size < 0:
            raise

        config['model_config']['validation_mode']['test_set_size'] = test_set_size

        return ''
    except:
        return 'Wprowadzono nieprawidłowy rozmiar zbioru treningowego. Wprowadź wartość z przedziału od 0 do 1'


@app.callback(Output('k-error', 'children'), [Input('k-input', 'value')])
def set_k_fold(value):
    try:
        if type(value) != int:
            raise

        config['model_config']['validation_mode']['k'] = value

        return ''
    except:
        return 'Wprowadzono nieprawidłową wartość k. Wprowadź liczbę całkowitą.'


@app.callback(Output('model-progress', 'children'),
              [Input('start-button-sgd', 'n_clicks')])
def model_progress_info(n_clicks):
    print('test')
    progress_info = html.P(id='progress-info', children='Trwa proces ucznia')

    return progress_info


@app.callback([Output('model-start', 'children'), Output('model-end', 'children')],
              [Input('progress-info', 'children')])
def click_start_button(children):
    print('epoch: ', config['model_config']['metrics']['n_epoch'])

    if children == 'Trwa proces ucznia':

        # global test
        #
        # test = False

        controller = ModelController()
        metrics_preprocessor = MetricPreprocessor()

        controller.run_model(config)

        raport_button = dbc.Row([html.Button(id='raport', children=[dcc.Link('Pokaż raport', href='/perceptron_sgd_raport')])],
                                justify='center',
                                style={'padding': '15px'})

        train_metrics, test_metrics = metrics_preprocessor.run_sgd(config['model_config'])

        # czy metrics_sgd jest mi faktycznie potrzebne?
        perceptron_sgd_raport.set_metrics(metrisc_sgd, train_metrics, test_metrics)

        if config['model_config']['validation_mode']['mode'] == 'simple_split':
            perceptron_sgd_raport.generate_raport()
        elif config['model_config']['validation_mode']['mode'] == 'cross_validation':
            perceptron_sgd_raport.generate_cv_raport()


        return raport_button, html.P(id='end-text', children='Zakończono!')
    else:
        pass








