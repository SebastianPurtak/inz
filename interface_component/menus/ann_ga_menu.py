import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from interface_component.app import app
from interface_component.raports import ann_ga_raport
from interface_component.utils.metrics_preprocessing import MetricPreprocessor
from interface_component.utils.data_management import DataManagement
from models_component.models_controller import ModelController

# ==CONFIG==============================================================================================================

config = {'model': 'ann_ga',  # perceptron_sgd
          'data_source': 'seed_data',  # seed_data
          'model_config': {'no_generations':        50,
                           'pop_size':              100,
                           'select_n':              0.3,
                           'mut_prob':              0.2,
                           'rand_mut':              0.5,
                           'mut_type':              'node_mut', # random_add
                           'n_hidden':              [5],
                           'selection_method':      'best_selection',
                           'parents_choice':        'random_parents',  # random_parents, sequence_parents
                            'cross_type':           'corss_nodes', # cross_uniform, cross_one_point, cross_two_point, corss_nodes
                           'max_fit':               0.1,
                           'evaluation_pop':        5,
                           'validation_mode': {'mode': 'simple_split',  # 'simple_split', 'cross_validation'
                                               'test_set_size': 0.25,
                                               'k': 10},
                           'metrics': {'data_train': [],
                                       'data_test': [],
                                       'cv_data_train': [],
                                       'cv_data_test': [],
                                       'n_epoch': [],
                                       'n_row': [],
                                       'prediction': [],
                                       'real_value': [],
                                       'error': [],
                                       'generation': [],
                                       'best_fit': [],
                                       'mean_fit':      [],
                                       'val_fit': []}}}


data_manager = DataManagement()
data_sources = data_manager.get_datasets_list()

validation_methods = ['simple_split', 'cross_validation']

mutation_type = ['swap_mut', 'random_mut']

selection_methods = ['best_selection', 'simple_selection']

parents_choice = ['sequence_parents', 'random_parents']

cross_type = ['cross_uniform', 'cross_one_point', 'cross_two_point', 'corss_nodes']

# ==LAYOUT==============================================================================================================

layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-ann-ga',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Sieć Neuronowa GA')],
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

    # ==OPCJE_KONFIGURACJI==============================================================================================
    dbc.Row([
        dbc.Col([
            # ==ŹRÓDŁO_DANYCH===========================================================================================
            dbc.Row(id='data_source-ann-ga-label',
                    children=[html.Label('Wybierz źródło danych')],
                    justify='center',
                    style={'padding': '10px'}),

            dbc.Row(id='data_source-ann-ga-choice',
                    children=[
                        dcc.Dropdown(id='data_source-ann_ga-input',
                                     options=[{'label': data_name, 'value': data_name} for data_name in data_sources],
                                     value=data_sources[0],
                                     clearable=False,
                                     style={'width': '200px'})],
                    justify='center'),

            dbc.Row(dbc.Col([dbc.Button(id='refresh-button-ann_ga', children='Odświerz', color='secondary', size='sm',
                                block=True)], width=2), justify='center', style={'padding': '10px'}),

            # ==LICZBA_POKOLEŃ==========================================================================================
            dbc.Row(id='no_generation-ann-ga-label',
                    children=[html.Label(('Wybierz liczbę pokoleń'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='no_generation-ann-ga-choice',
                    children=[
                        dcc.Input(id='no_generation-ann-ga-input',
                                  value=50,
                                  type='number',
                                  min=0,
                                  style={'width': '200px'})],
                    justify="center"),

            # ==WIELKOŚĆ_POPULACJI======================================================================================
            dbc.Row(id='pop_size-ann-ga-label',
                    children=[html.Label(('Wybierz liczebność populacji'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='pop_size-ann-ga-choice',
                    children=[
                        dcc.Input(id='pop_size-ann-ga-input',
                                  value=10,
                                  type='number',
                                  min=0,
                                  style={'width': '200px'})],
                    justify="center"),

            # ==SELECT_N================================================================================================

            dbc.Row(id='select_n-ann-ga-label',
                    children=[html.Label(('Wybierz wspólczynnik selekcji'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='select_n-ann-ga-choice',
                    children=[
                        dcc.Input(id='select_n-ann-ga-input',
                                  value=0.3,
                                  type='text',
                                  min=0,
                                  style={'width': '200px'})],
                    justify='center'),

            # ==MUT_TYPE================================================================================================

            dbc.Row(id='mut_type-ann-ga-label',
                    children=[html.Label('Wybierz rodzaj mutacji')],
                    justify='center',
                    style={'padding': '10px'}),

            dbc.Row(id='mut_type-ann-ga-choice',
                    children=[
                        dcc.Dropdown(id='mut_type-ann-ga-input',
                                     options=[{'label': mut_type, 'value': mut_type} for mut_type in mutation_type],
                                     value=mutation_type[0],
                                     clearable=False,
                                     style={'width': '200px'})],
                    justify='center'),

            # ==MUT_PROB================================================================================================

            dbc.Row(id='mut_prob-ann-ga-label',
                    children=[html.Label(('Wybierz wspólczynnik mutacji genomu'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='mut_prob-ann-ga-choice',
                    children=[
                        dcc.Input(id='mut_prob-ann-ga-input',
                                  value=0.2,
                                  type='text',
                                  min=0,
                                  style={'width': '200px'})],
                    justify='center'),

        ]),

        dbc.Col([

            # ==RAND_MUT================================================================================================

            dbc.Row(id='rand_mut-ann-ga-label',
                    children=[html.Label(('Wybierz wspólczynnik mutacji genu'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='rand_mut-ann-ga-choice',
                    children=[
                        dcc.Input(id='rand_mut-ann-ga-input',
                                  value=0.1,
                                  type='text',
                                  min=0,
                                  style={'width': '200px'})],
                    justify='center'),

            # ==METODA_SELEKCJI=========================================================================================
            dbc.Row(id='selection_method-ann-ga-label',
                    children=[html.Label(('Wybierz metodę selekcji'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='selection_method-ann-ga-choice',
                    children=[
                        dcc.Dropdown(id='selection_method-ann-ga-input',
                                     options=[{'label': select_meth, 'value': select_meth}
                                              for select_meth in selection_methods],
                                     value=selection_methods[0],
                                     clearable=False,
                                     style={'width': '200px'})],
                    justify='center'),

            # ==PARENTS_CHOICE==========================================================================================
            dbc.Row(id='parents_choice-ann-ga-label',
                    children=[html.Label(('Wybierz metodę doboru rodziców'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='parents_choice-ann-ga-choice',
                    children=[
                        dcc.Dropdown(id='parents_choice-ann-ga-input',
                                     options=[{'label': p_choice, 'value': p_choice} for p_choice in parents_choice],
                                     value=parents_choice[0],
                                     clearable=False,
                                     style={'width': '200px'})],
                    justify='center'),

            # ==CROSS_TYPE==============================================================================================

            dbc.Row(id='cross_type-ann-ga-label',
                    children=[html.Label(('Wybierz metodę krzyżowania'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='cross_type-ann-ga-choice',
                    children=[
                        dcc.Dropdown(id='cross_type-ann-ga-input',
                                     options=[{'label': c_type, 'value': c_type} for c_type in cross_type],
                                     value=cross_type[0],
                                     clearable=False,
                                     style={'width': '200px'})],
                    justify='center'),

            # ==EVELUATION_POP==========================================================================================

            dbc.Row(id='eva_pop-ann-ga-label',
                    children=[html.Label(('Wybierz wielkość populacji testowej'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='eva_pop-ann-ga-choice',
                    children=[
                        dcc.Input(id='eva_pop-ann-ga-input',
                                  value=5,
                                  type='number',
                                  min=0,
                                  style={'width': '200px'})],
                    justify="center"),

            # ==TEST_SET_SIZE===========================================================================================

            dbc.Row(id='test_set_size-ann-ga-label',
                    children=[html.Label(('Wybierz wspólczynnik podziału na zbiór treningowy i testowy'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='test_set_size-ann-ga-choice',
                    children=[
                        dcc.Input(id='test_set_size-ann-ga-input',
                                  value=0.25,
                                  type='text',
                                  min=0,
                                  style={'width': '200px'})],
                    justify='center'),

            # ==MAX_FIT=================================================================================================

            dbc.Row(id='max_fit-ann-ga-label',
                    children=[html.Label(('Wybierz minimalną wartość funkcji dopasowania'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='max_fit-ann-ga-choice',
                    children=[
                        dcc.Input(id='max_fit-ann-ga-input',
                                  value=0.1,
                                  type='text',
                                  min=0,
                                  style={'width': '200px'})],
                    justify='center'),
        ])

    ], justify='center'),

    # ==PODGLĄD=========================================================================================================

    dbc.Row(id='start-button-ann-ga-row', children=html.Button(id='start-button-ann-ga', children='Start'),
            style={'padding': '10px'}, justify='center'),

    dbc.Row(html.Label('Komunikaty:'), justify='center'),

    dbc.Row(id='start-model-info-row-ann-ga', children=[], justify='center'),
    dbc.Row(id='end-model-info-row-ann-ga', children=[], justify='center'),
    dbc.Row(id='data_source-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='data_refresh-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='no_generations-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='pop_size-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='select_n-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='mut_type-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='mut_prob-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='rand_mut-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='selection_method-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='parents_choice-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='cross_type-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='evaluation_pop-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='test_set_size-pga-alert-ann-ga', children=[], justify='center'),
    dbc.Row(id='max_fit-pga-alert-ann-ga', children=[], justify='center'),

    dbc.Row(id='raport-button-row-ann-ga', children=[], justify='center', style={'padding': '15px'}),
    dbc.Row([html.Button(id='back-ann-ga', children=[dcc.Link('Wróć', href='/models')])],
            justify='center',
            style={'padding': '15px'}),

], fluid=True,
    style={'backgroundColor': '#D3D3D3'})

# ==CALLBACKS===========================================================================================================


@app.callback(Output('data_source-pga-alert-ann-ga', 'children'), [Input('data_source-ann_ga-input', 'value')])
def set_data_source_ann_ga(value):
    config['data_source'] = value


@app.callback([Output('data_source-ann-ga-choice', 'children'), Output('data_refresh-pga-alert-ann-ga', 'children')],
              [Input('refresh-button-ann_ga', 'n_clicks')])
def set_ann_ga_data_source(n_clicks):
    global data_sources
    data_sources = data_manager.get_datasets_list()

    drop_menu = dcc.Dropdown(id='data_source-ann_ga-input',
                             options=[{'label': data_name, 'value': data_name} for data_name in data_sources],
                             clearable=False,
                             value=data_sources[0],
                             style={'width': '200px'})

    return drop_menu, None


@app.callback(Output('no_generations-pga-alert-ann-ga', 'children'), [Input('no_generation-ann-ga-input', 'value')])
def set_no_generations_ann_ga(value):
    try:
        if value <= 0 or type(value) != int:
            return dbc.Alert(id='no_generations_0_alert',
                             children='Liczba popkoleń powinna być liczbą całkowitą większą od 0.',
                             color='danger')

        config['model_config']['no_generations'] = value
    except:
        return dbc.Alert(id='no_generations_None_alert', children='Podaj liczbę pokolej!', color='danger')


@app.callback(Output('pop_size-pga-alert-ann-ga', 'children'), [Input('pop_size-ann-ga-input', 'value')])
def set_pop_size_ann_ga(value):
    try:
        if value <= 0 or type(value) != int:
            return dbc.Alert(id='pop_size_0_alert',
                             children='Liczebność populacji powinna być liczbą całkowitą większą od 0.',
                             color='danger')

        config['model_config']['pop_size'] = value
    except:
        return dbc.Alert(id='pop_size_None_alert', children='Podaj liczebność populacji!', color='danger')


@app.callback(Output('select_n-pga-alert-ann-ga', 'children'), [Input('select_n-ann-ga-input', 'value')])
def set_select_n_ann_ga(select_n_value):
    try:
        if float(select_n_value) > 1:
            return dbc.Row(
                dbc.Alert('Współczynnik selekcji musi mieścić się w przedziale od 0 do 1!', color='danger'),
                justify='center')

        config['model_config']['select_n'] = float(select_n_value)

    except:
        return dbc.Row(
            dbc.Alert('Współczynnik selekcji musi mieścić się w przedziale od 0 do 1!', color='danger'),
            justify='center')


@app.callback(Output('mut_type-pga-alert-ann-ga', 'children'), [Input('mut_type-ann-ga-input', 'value')])
def set_mut_type_ann_ga(value):
    config['model_config']['mut_type'] = value


@app.callback(Output('mut_prob-pga-alert-ann-ga', 'children'), [Input('mut_prob-ann-ga-input', 'value')])
def set_mut_prob_ann_ga(mut_prob_value):
    try:
        if float(mut_prob_value) > 1:
            return dbc.Row(
                dbc.Alert('Współczynnik mutacji genomu musi mieścić się w przedziale od 0 do 1!', color='danger'),
                justify='center')

        config['model_config']['mut_prob'] = float(mut_prob_value)

    except:
        return dbc.Row(
            dbc.Alert('Współczynnik mutacji genomu musi mieścić się w przedziale od 0 do 1!', color='danger'),
            justify='center')


@app.callback(Output('rand_mut-pga-alert-ann-ga', 'children'), [Input('rand_mut-ann-ga-input', 'value')])
def set_rand_mut_ann_ga(rand_mut_value):
    try:
        if float(rand_mut_value) > 1:
            return dbc.Row(
                dbc.Alert('Współczynnik mutacji genu musi mieścić się w przedziale od 0 do 1!', color='danger'),
                justify='center')

        config['model_config']['rand_mut'] = float(rand_mut_value)

    except:
        return dbc.Row(
            dbc.Alert('Współczynnik mutacji genu musi mieścić się w przedziale od 0 do 1!', color='danger'),
            justify='center')


@app.callback(Output('selection_method-pga-alert-ann-ga', 'children'), [Input('selection_method-ann-ga-input', 'value')])
def set_selection_method_ann_ga(value):
    config['model_config']['selection_method'] = value


@app.callback(Output('parents_choice-pga-alert-ann-ga', 'children'), [Input('parents_choice-ann-ga-input', 'value')])
def set_parent_choice_ann_ga(value):
    config['model_config']['parents_choice'] = value


@app.callback(Output('cross_type-pga-alert-ann-ga', 'children'), [Input('cross_type-ann-ga-input', 'value')])
def set_cross_type_ann_ga(value):
    config['model_config']['cross_type'] = value


@app.callback(Output('evaluation_pop-pga-alert-ann-ga', 'children'), [Input('eva_pop-ann-ga-input', 'value')])
def set_evaluation_pop_ann_ga(value):
    config['model_config']['evaluation_pop'] = value


@app.callback(Output('test_set_size-pga-alert-ann-ga', 'children'), [Input('test_set_size-ann-ga-input', 'value')])
def set_test_set_size_pga(test_set_size_value):
    try:
        if float(test_set_size_value) > 1:
            return dbc.Row(
                dbc.Alert('Współczynnik podziału na zbiór treningowy i testowy musi mieścić się w przedziale od 0 do 1!',
                          color='danger'),
                justify='center')

        config['model_config']['validation_mode']['test_set_size'] = float(test_set_size_value)
        # print('config value: ', type(config['model_config']['validation_mode']['test_set_size']))

    except:
        return dbc.Row(
            dbc.Alert('Współczynnik podziału na zbiór treningowy i testowy musi mieścić się w przedziale od 0 do 1!',
                      color='danger'),
            justify='center')


@app.callback(Output('max_fit-pga-alert-ann-ga', 'children'), [Input('max_fit-ann-ga-input', 'value')])
def set_max_fit_ann_ga(max_fit_value):
    try:
        if float(max_fit_value) > 1:
            return dbc.Row(
                dbc.Alert('Minimalna wartość funkcji dopasowania musi mieścić się w przedziale od 0 do 1!',
                          color='danger'),
                justify='center')

        config['model_config']['max_fit'] = float(max_fit_value)

    except:
        return dbc.Row(
            dbc.Alert('Minimalna wartość funkcji dopasowania musi mieścić się w przedziale od 0 do 1!',
                      color='danger'),
            justify='center')


@app.callback(Output('start-model-info-row-ann-ga', 'children'), [Input('start-button-ann-ga', 'n_clicks')])
def click_start_ann_ga_model(n_clicks):
    if n_clicks is not None:
        return dbc.Alert(id='progress-ann-ga-info', children='Trwa proces uczenia', color='primary')


@app.callback([Output('end-model-info-row-ann-ga', 'children'), Output('raport-button-row-ann-ga', 'children')],
              [Input('progress-ann-ga-info', 'children')])
def run_ann_ga_model(children):
    controller = ModelController()
    metrics_preprocessor = MetricPreprocessor()

    # print('generations: ', config['model_config']['metrics']['generation'])

    controller.run_model(config)

    raport_button = dbc.Row(
        [html.Button(id='raport-pga', children=[dcc.Link('Pokaż raport', href='/models/ann_ga_raport')])],
        justify='center',
        style={'padding': '15px'})

    train_metrics, test_metrics = metrics_preprocessor.perprocess_ga_metrics(config['model_config'])

    ann_ga_raport.generate_raport('/models/ann_ga_menu', train_metrics, test_metrics)
    # perceptron_ga_raport.generate_raport(train_metrics, test_metrics)

    return dbc.Alert(id='result-ann-ga-info', children='Zakończono!', color='primary'), raport_button