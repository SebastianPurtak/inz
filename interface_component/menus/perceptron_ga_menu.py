import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from interface_component.app import app
from interface_component.raports import perceptron_ga_raport
from interface_component.utils.metrics_preprocessing import MetricPreprocessor
from interface_component.utils.data_management import DataManagement
from models_component.models_controller import ModelController

# ==CONFIG==============================================================================================================

config = {'model': 'perceptron_ga',
              'data_source': 'sonar_data',
              'model_config': {'no_generations':    50,
                               'pop_size':          10,
                               'select_n':          0.3,
                               'mut_prob':          0.2,
                               'rand_mut':          0.1,
                               'mut_type':          'swap_mut',
                               'selection_method':  'simple_selection',
                               'parents_choice':    'sequence_parents',
                               'cross_type':        'cross_uniform',
                               'evaluation_pop':     5,
                               'max_fit':           0.1,
                               'validation_mode': {'mode': 'simple_split',
                                                   'test_set_size': 0.25,
                                                   'k': 0},
                               'metrics': {'data_train':    [],
                                           'data_test':     [],
                                           'cv_data_train': [],
                                           'cv_data_test':  [],
                                           'n_epoch':       [],
                                           'n_row':         [],
                                           'prediction':    [],
                                           'real_value':    [],
                                           'error':         [],
                                           'generation':    [],
                                           'best_fit':      [],
                                           'mean_fit':      [],
                                           'val_fit':       []}}}


data_manager = DataManagement()
data_sources = data_manager.get_datasets_list()

validation_methods = ['cross_validation', 'simple_split']

mutation_type = ['swap_mut', 'random_mut']

selection_methods = ['best_selection', 'simple_selection']

parents_choice = ['sequence_parents', 'random_parents']

cross_type = ['cross_uniform', 'cross_one_point', 'cross_two_point']

# ==LAYOUT==============================================================================================================

layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-pga',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Perceptron GA')],
                        style={
                            'position': 'relative',
                            'top': '20%',
                            'textAlign': 'center',
                            'margin-bottom': '75px'
                        })
                ])
            ]),

    # ==OPCJE_KONFIGURACJI==============================================================================================

    dbc.Row([
        dbc.Col([
            # ==ŹRÓDŁO_DANYCH===========================================================================================
            dbc.Row(id='data_source-pga-label',
                    children=[html.Label('Wybierz źródło danych')],
                    justify='center',
                    style={'padding': '10px'}),

            dbc.Row(id='data_source-pga-choice',
                    children=[
                        dcc.Dropdown(id='data_source-pga-input',
                                     options=[{'label': data_name, 'value': data_name} for data_name in data_sources],
                                     value=data_sources[0],
                                     clearable=False,
                                     style={'width': '200px', 'color': '#000000'})],
                    justify='center'),

            dbc.Row(dbc.Col([dbc.Button(id='refresh-button-pga', children='Odświerz', color='secondary', size='sm',
                                block=True)], width=2), justify='center',
                    style={'padding': '10px', 'margin-bottom': '15px'}),

            # ==LICZBA_POKOLEŃ==========================================================================================
            dbc.Row(id='no_generation-pga-label',
                    children=[html.Label(('Wybierz liczbę pokoleń'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='no_generation-pga-choice',
                    children=[
                        dcc.Input(id='no_generation-pga-input',
                                  value=50,
                                  type='number',
                                  min=0,
                                  style={'width': '200px'})],
                    justify="center", style={'margin-bottom': '40px'}),

            # ==WIELKOŚĆ_POPULACJI======================================================================================
            dbc.Row(id='pop_size-pga-label',
                    children=[html.Label(('Wybierz liczebność populacji'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='pop_size-pga-choice',
                    children=[
                        dcc.Input(id='pop_size-pga-input',
                                  value=10,
                                  type='number',
                                  min=0,
                                  style={'width': '200px'})],
                    justify="center", style={'margin-bottom': '40px'}),

            # ==SELECT_N================================================================================================

            dbc.Row(id='select_n-pga-label',
                    children=[html.Label(('Wybierz wspólczynnik selekcji'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='select_n-pga-choice',
                    children=[
                        dcc.Input(id='select_n-pga-input',
                                  value=0.3,
                                  type='text',
                                  min=0,
                                  style={'width': '200px'})],
                    justify='center', style={'margin-bottom': '40px'}),

            # ==MUT_TYPE================================================================================================

            dbc.Row(id='mut_type-pga-label',
                    children=[html.Label('Wybierz rodzaj mutacji')],
                    justify='center',
                    style={'padding': '10px'}),

            dbc.Row(id='mut_type-pga-choice',
                    children=[
                        dcc.Dropdown(id='mut_type-pga-input',
                                     options=[{'label': mut_type, 'value': mut_type} for mut_type in mutation_type],
                                     value=mutation_type[0],
                                     clearable=False,
                                     style={'width': '200px', 'color': '#000000'})],
                    justify='center', style={'margin-bottom': '40px'}),

            # ==MUT_PROB================================================================================================

            dbc.Row(id='mut_prob-pga-label',
                    children=[html.Label(('Wybierz wspólczynnik mutacji genomu'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='mut_prob-pga-choice',
                    children=[
                        dcc.Input(id='mut_prob-pga-input',
                                  value=0.2,
                                  type='text',
                                  min=0,
                                  style={'width': '200px'})],
                    justify='center', style={'margin-bottom': '40px'}),
        ]),

        dbc.Col([

            # ==RAND_MUT================================================================================================

            dbc.Row(id='rand_mut-pga-label',
                    children=[html.Label(('Wybierz wspólczynnik mutacji genu'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='rand_mut-pga-choice',
                    children=[
                        dcc.Input(id='rand_mut-pga-input',
                                  value=0.1,
                                  type='text',
                                  min=0,
                                  style={'width': '200px'})],
                    justify='center', style={'margin-bottom': '65px'}),

            # ==METODA_SELEKCJI=========================================================================================
            dbc.Row(id='selection_method-pga-label',
                    children=[html.Label(('Wybierz metodę selekcji'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='selection_method-pga-choice',
                    children=[
                        dcc.Dropdown(id='selection_method-pga-input',
                                     options=[{'label': select_meth, 'value': select_meth}
                                              for select_meth in selection_methods],
                                     value=selection_methods[0],
                                     clearable=False,
                                     style={'width': '200px', 'color': '#000000'})],
                    justify='center', style={'margin-bottom': '40px'}),

            # ==PARENTS_CHOICE==========================================================================================
            dbc.Row(id='parents_choice-pga-label',
                    children=[html.Label(('Wybierz metodę doboru rodziców'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='parents_choice-pga-choice',
                    children=[
                        dcc.Dropdown(id='parents_choice-pga-input',
                                     options=[{'label': p_choice, 'value': p_choice} for p_choice in parents_choice],
                                     value=parents_choice[0],
                                     clearable=False,
                                     style={'width': '200px', 'color': '#000000'})],
                    justify='center', style={'margin-bottom': '40px'}),

            # ==CROSS_TYPE==============================================================================================

            dbc.Row(id='cross_type-pga-label',
                    children=[html.Label(('Wybierz metodę krzyżowania'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='cross_type-pga-choice',
                    children=[
                        dcc.Dropdown(id='cross_type-pga-input',
                                     options=[{'label': c_type, 'value': c_type} for c_type in cross_type],
                                     value=cross_type[0],
                                     clearable=False,
                                     style={'width': '200px', 'color': '#000000'})],
                    justify='center', style={'margin-bottom': '40px'}),

            # ==EVELUATION_POP==========================================================================================

            dbc.Row(id='eva_pop-pga-label',
                    children=[html.Label(('Wybierz wielkość populacji testowej'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='eva_pop-pga-choice',
                    children=[
                        dcc.Input(id='eva_pop-pga-input',
                                  value=5,
                                  type='number',
                                  min=0,
                                  style={'width': '200px', 'color': '#000000'})],
                    justify="center", style={'margin-bottom': '40px'}),

            # ==TEST_SET_SIZE===========================================================================================

            dbc.Row(id='test_set_size-pga-label',
                    children=[html.Label(('Wybierz wspólczynnik podziału na zbiór treningowy i testowy'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='test_set_size-pga-choice',
                    children=[
                        dcc.Input(id='test_set_size-pga-input',
                                  value=0.25,
                                  type='text',
                                  min=0,
                                  style={'width': '200px'})],
                    justify='center', style={'margin-bottom': '40px'}),

            # ==MAX_FIT=================================================================================================

            dbc.Row(id='max_fit-pga-label',
                    children=[html.Label(('Wybierz minimalną wartość funkcji dopasowania'))],
                    justify="center",
                    style={'padding': '10px'}),

            dbc.Row(id='max_fit-pga-choice',
                    children=[
                        dcc.Input(id='max_fit-pga-input',
                                  value=0.1,
                                  type='text',
                                  min=0,
                                  style={'width': '200px'})],
                    justify='center', style={'margin-bottom': '40px'}),
        ])

    ], justify='center'),

    # ==PODGLĄD=========================================================================================================

    dbc.Row([dbc.Col([dbc.Button(id='start-button-pga', children='Start', color='secondary', size='sm',
                                 block=True)], width=2)],
            justify='center', style={'padding': '15px'}),

    dbc.Row(html.Label('Komunikaty:'), justify='center'),

    dbc.Row(id='start-model-info-row', children=[], justify='center'),
    dbc.Row(id='end-model-info-row', children=[], justify='center'),
    dbc.Row(id='data_source-pga-alert', children=[], justify='center'),
    dbc.Row(id='data_refresh-pga-alert', children=[], justify='center'),
    dbc.Row(id='no_generations-pga-alert', children=[], justify='center'),
    dbc.Row(id='pop_size-pga-alert', children=[], justify='center'),
    dbc.Row(id='select_n-pga-alert', children=[], justify='center'),
    dbc.Row(id='mut_type-pga-alert', children=[], justify='center'),
    dbc.Row(id='mut_prob-pga-alert', children=[], justify='center'),
    dbc.Row(id='rand_mut-pga-alert', children=[], justify='center'),
    dbc.Row(id='selection_method-pga-alert', children=[], justify='center'),
    dbc.Row(id='parents_choice-pga-alert', children=[], justify='center'),
    dbc.Row(id='cross_type-pga-alert', children=[], justify='center'),
    dbc.Row(id='evaluation_pop-pga-alert', children=[], justify='center'),
    dbc.Row(id='test_set_size-pga-alert', children=[], justify='center'),
    dbc.Row(id='max_fit-pga-alert', children=[], justify='center'),


    dbc.Row(id='raport-button-row', children=[], justify='center', style={'padding': '15px'}),

    dbc.Row([dbc.Col([dbc.Button(id='back_pga', children='Wróć', color='secondary', href='/models',
                                 size='sm', block=True)], width=2)],
            justify='center', style={'padding': '15px', 'margin-bottom': '20px'}),

],
    fluid=True)


# ==CALLBACKS===========================================================================================================


@app.callback(Output('data_source-pga-alert', 'children'), [Input('data_source-pga-input', 'value')])
def set_data_source_pga(value):
    config['data_source'] = value


@app.callback([Output('data_source-pga-choice', 'children'), Output('data_refresh-pga-alert', 'children')],
              [Input('refresh-button-pga', 'n_clicks')])
def set_pga_data_source(n_clicks):
    global data_sources
    data_sources = data_manager.get_datasets_list()
    data_sources.remove('.gitkeep')

    drop_menu = dcc.Dropdown(id='data_source-pga-input',
                             options=[{'label': data_name, 'value': data_name} for data_name in data_sources],
                             clearable=False,
                             value=data_sources[0],
                             style={'width': '200px', 'color': '#000000'})

    return drop_menu, None


@app.callback(Output('no_generations-pga-alert', 'children'), [Input('no_generation-pga-input', 'value')])
def set_no_generations_pga(value):
    try:
        if value <= 0 or type(value) != int:
            return dbc.Alert(id='no_generations_0_alert',
                             children='Liczba popkoleń powinna być liczbą całkowitą większą od 0.',
                             color='danger')

        config['model_config']['no_generations'] = value
    except:
        return dbc.Alert(id='no_generations_None_alert', children='Podaj liczbę pokolej!', color='danger')


@app.callback(Output('pop_size-pga-alert', 'children'), [Input('pop_size-pga-input', 'value')])
def set_pop_size_pga(value):
    try:
        if value <= 0 or type(value) != int:
            return dbc.Alert(id='pop_size_0_alert',
                             children='Liczebność populacji powinna być liczbą całkowitą większą od 0.',
                             color='danger')

        config['model_config']['pop_size'] = value
    except:
        return dbc.Alert(id='pop_size_None_alert', children='Podaj liczebność populacji!', color='danger')


@app.callback(Output('select_n-pga-alert', 'children'), [Input('select_n-pga-input', 'value')])
def set_select_n_pga(select_n_value):
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


@app.callback(Output('mut_type-pga-alert', 'children'), [Input('mut_type-pga-input', 'value')])
def set_mut_type_pga(value):
    config['model_config']['mut_type'] = value


@app.callback(Output('mut_prob-pga-alert', 'children'), [Input('mut_prob-pga-input', 'value')])
def set_mut_prob_pga(mut_prob_value):
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


@app.callback(Output('rand_mut-pga-alert', 'children'), [Input('rand_mut-pga-input', 'value')])
def set_rand_mut_pga(rand_mut_value):
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


@app.callback(Output('selection_method-pga-alert', 'children'), [Input('selection_method-pga-input', 'value')])
def set_selection_method_pga(value):
    config['model_config']['selection_method'] = value


@app.callback(Output('parents_choice-pga-alert', 'children'), [Input('parents_choice-pga-input', 'value')])
def set_parent_choice_pga(value):
    config['model_config']['parents_choice'] = value


@app.callback(Output('cross_type-pga-alert', 'children'), [Input('cross_type-pga-input', 'value')])
def set_cross_type_pga(value):
    config['model_config']['cross_type'] = value


@app.callback(Output('evaluation_pop-pga-alert', 'children'), [Input('eva_pop-pga-input', 'value')])
def set_evaluation_pop_pga(value):
    config['model_config']['evaluation_pop'] = value


@app.callback(Output('test_set_size-pga-alert', 'children'), [Input('test_set_size-pga-input', 'value')])
def set_test_set_size_pga(test_set_size_value):
    try:
        if float(test_set_size_value) > 1:
            return dbc.Row(
                dbc.Alert('Współczynnik podziału na zbiór treningowy i testowy musi mieścić się w przedziale od 0 do 1!',
                          color='danger'),
                justify='center')

        config['model_config']['validation_mode']['test_set_size'] = float(test_set_size_value)

    except:
        return dbc.Row(
            dbc.Alert('Współczynnik podziału na zbiór treningowy i testowy musi mieścić się w przedziale od 0 do 1!',
                      color='danger'),
            justify='center')


@app.callback(Output('max_fit-pga-alert', 'children'), [Input('max_fit-pga-input', 'value')])
def set_max_fit_pga(max_fit_value):
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


@app.callback(Output('start-model-info-row', 'children'), [Input('start-button-pga', 'n_clicks')])
def click_start_pga_model(n_clicks):
    if n_clicks is not None:
        return dbc.Alert(id='progress-pga-info', children='Trwa proces uczenia', color='primary')


@app.callback([Output('end-model-info-row', 'children'), Output('raport-button-row', 'children')],
              [Input('progress-pga-info', 'children')])
def run_pga_model(children):
    controller = ModelController()
    metrics_preprocessor = MetricPreprocessor()

    controller.run_model(config)

    raport_button = dbc.Col([dbc.Button(id='raport-psgd', children='Pokaż raport', color='secondary',
                                        href='/models/perceptron_ga_raport', size='sm', block=True)],
                            width=2)

    train_metrics, test_metrics = metrics_preprocessor.perprocess_ga_metrics(config['model_config'])

    perceptron_ga_raport.generate_raport('/models/perceptron_ga_menu', train_metrics, test_metrics)

    return dbc.Alert(id='result-pga-info', children='Zakończono!', color='primary'), raport_button




