import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from interface_component.app import app
from interface_component.utils.db_facade import DBFacade
from interface_component.utils.raport_exporter import RaportExporter
from interface_component.raports import perceptron_sgd_raport, perceptron_ga_raport, ann_bp_raport, ann_ga_raport

db_facade = DBFacade()
collections_list = db_facade.get_collections_list()


layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-results-menu',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Podgląd wyników')],
                        style={
                            'position': 'relative',
                            'top': '20%',
                            'textAlign': 'center',
                            'margin-bottom': '100px'
                        })
                ])
            ]),

    # ==OPCJE_PODGLĄDU_WYNIKÓW==========================================================================================

    dbc.Row(html.H3('Wczytaj raport z bazy danych'), justify='center',
            style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row(html.H4('Wybierz model'), justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row(id='raports-list-choice-row', children=[
        dcc.Dropdown(id='raports-list-choice',
                     options=[
                         {'label': value, 'value': value} for value in collections_list
                     ],
                     clearable=False,
                     value=collections_list[0],
                     style={'width': '300px', 'color': '#000000'})
    ],
            justify='center',
            style={'padding': '10px', 'margin-bottom': '10px'}),


    dbc.Row(html.H4('Zapisane raporty:'), justify='center', style={'padding': '10px', 'margin-bottom': '10px'}),

    dbc.Row(dbc.Col(id='raports-preview', children=[], width=8), justify='center',
            style={'padding': '10px', 'margin-bottom': '10px'}),

    dbc.Row(html.H4('Wybierz raport:'), justify='center', style={'padding': '10px', 'margin-bottom': '10px'}),

    dbc.Row(id='raports_list-psgd', children=[dcc.Dropdown(id='raports-choice',
                     options=[{'label': value, 'value': value} for value in {'enpty': 'enpty'}],
                     clearable=False,
                     value='empty',
                     style={'width': '300px', 'color': '#000000'})], justify='center',
            style={'padding': '10px', 'margin-bottom': '10px'}),

    dbc.Row(id='load_button_row', children=[], justify='center', style={'padding': '10px', 'margin-bottom': '10px'}),

    dbc.Row(dbc.Col(children=[dbc.Button(id='delete-raport-button', children='Usuń raport', color='secondary',
                                         size='lg', block=True)], width=4), justify='center',
            style={'padding': '10px', 'margin-bottom': '20px'}),

    # ==WCZYTYWANIE_Z_JSON==============================================================================================

    dbc.Row(html.H3('Wczytaj raport z pliku json'), justify='center', style={'padding': '10px', 'margin-bottom': '20px'}),

    dbc.Row(dbc.Col([dcc.Upload(id='upload-json-data', children=html.Div(['Przeciągnij albo ', html.A('wskaż plik')]),
                                style={'width': '100%',
                                       'height': '60px',
                                       'lineHeight': '60px',
                                       'borderWidth': '1px',
                                       'borderStyle': 'dashed',
                                       'borderRadius': '5px',
                                       'textAlign': 'center',
                                       'margin': '10px'})], width=4), justify='center',
            style={'padding': '10px', 'margin-bottom': '20px'}),


    # ==KOMUNIKATY_BŁĘDÓW===============================================================================================

    dbc.Row(id='lists-raports-psgd-alert', children=[], justify='center'),
    dbc.Row(id='lists-raports-pga-alert', children=[], justify='center'),
    dbc.Row(id='lists-raports-ann-bp-alert', children=[], justify='center'),
    dbc.Row(id='lists-raports-ann-ga-alert', children=[], justify='center'),
    dbc.Row(id='read-raport-alert', children=[], justify='center'),
    dbc.Row(id='choice-raport-alert', children=[], justify='center'),
    dbc.Row(id='delete-raport-alert', children=[], justify='center'),
    dbc.Row(id='load-raport-alert', children=[], justify='center'),

    dbc.Row([dbc.Col([dbc.Button('Wróć', color='secondary', href='/',
                                         size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px'}),


], fluid=True)


# ==ZAPISYWANIE W BAZIE=================================================================================================

@app.callback([Output('raports-preview', 'children'), Output('raports_list-psgd', 'children')],
              [Input('raports-list-choice', 'value')])
def show_collections_list(collection_name):
    if collection_name is None:
        collection_name = db_facade.get_collections_list()[0]
    raports_list = db_facade.get_raport_list(collection_name)

    table = dbc.Row(id='collections-list-table', children=[
        html.Table([
            html.Tbody([
                html.Tr(html.Td(str(collection), style={'text-align': 'center'})) for collection in raports_list
            ])
        ])
    ], justify='center'),

    itable = dbc.Row(id='raports-choice-row', children=[
        dcc.Dropdown(id='raports-choice',
                     options=[{'label': value, 'value': value} for value in raports_list],
                     clearable=False,
                     value=raports_list[0],
                     style={'width': '300px'})],
                     justify='center',
                     style={'padding': '10px', 'color': '#000000'}),

    return table, itable


@app.callback(Output('load_button_row', 'children'),
              [Input('raports-choice', 'value')])
def choice_raport(value):
    if 'perceptron_sgd' in value:
        # print(value)
        test_metrics, train_metrics = db_facade.get_raport_data(value)

        button = generate_perceptron_sgd_raport(train_metrics, test_metrics)

        return button

        # perceptron_sgd_raport.set_metrics(train_metrics, test_metrics)
        # if isinstance(train_metrics, list):
        #     perceptron_sgd_raport.generate_cv_raport('/results_menu')
        # else:
        #     perceptron_sgd_raport.generate_raport('/results_menu')
        #
        # return dbc.Col(children=[dbc.Button(id='load-raport-button', children='Wczytaj raport', color='secondary',
        #                                 size='lg', block=True, href='/models/perceptron_sgd_raport')], width=4)

    elif 'perceptron_ga' in value:
        test_metrics, train_metrics = db_facade.get_raport_data(value)

        button = generate_perceptron_ga_raport(train_metrics, test_metrics)

        return button


        # perceptron_ga_raport.generate_raport('/results_menu', train_metrics, test_metrics)
        #
        # return dbc.Col(children=[dbc.Button(id='load-raport-button', children='Wczytaj raport', color='secondary',
        #                                     size='lg', block=True, href='/models/perceptron_ga_raport')], width=4)

    elif 'ann_bp' in value:
        test_metrics, train_metrics = db_facade.get_raport_data(value)

        button = generate_ann_bp_raport(train_metrics, test_metrics)
        return button

        # ann_bp_raport.set_metrics(train_metrics, test_metrics)
        # if isinstance(train_metrics, list):
        #     ann_bp_raport.generate_ann_bp_cv_raport('/results_menu')
        # else:
        #     ann_bp_raport.generate_ann_bp_split_raport('/results_menu')
        #
        # return dbc.Col(children=[dbc.Button(id='load-raport-button', children='Wczytaj raport', color='secondary',
        #                                     size='lg', block=True, href='/models/ann-bp_raport')], width=4)

    elif 'ann_ga' in value:
        test_metrics, train_metrics = db_facade.get_raport_data(value)

        button = generate_ann_ga_raport(train_metrics, test_metrics)
        return button

        # ann_ga_raport.generate_raport('/results_menu', train_metrics, test_metrics)
        #
        # return dbc.Col(children=[dbc.Button(id='load-raport-button', children='Wczytaj raport', color='secondary',
        #                                     size='lg', block=True, href='/models/ann_ga_raport')], width=4)


@app.callback(Output('delete-raport-alert', 'children'), [Input('delete-raport-button', 'n_clicks')])
def delete_raport(n_clicks):
    if n_clicks is not None:
        raport_name = db_facade.delete_raport()

        if raport_name != 'Brak połączenia z bazą danych':
            return dbc.Alert(id='delete-info', children=f'Usunięto {raport_name}', color='danger')
        else:
            return dbc.Alert(id='delete-info', children=f'{raport_name}', color='danger')


# ==ZAPISYWANIE W JSON==================================================================================================

@app.callback(Output('load-raport-alert', 'children'), [Input('upload-json-data', 'contents')],
              [State('upload-json-data', 'filename'), State('upload-json-data', 'last_modified')])
def upload_json_file(data, list_of_names, list_of_dates):
    if data is not None:
        exporter = RaportExporter()
        try:
            raport_type, test_metrics, train_metrics = exporter.from_json(data)
        except:
            return dbc.Alert(id='delete-info', children='Nie udało się wczytać pliku', color='danger')

        # Wybierz typ generowania raportów
        raport_generator = {'perceptron_sgd':   generate_perceptron_sgd_raport,
                            'perceptron_ga':    generate_perceptron_ga_raport,
                            'ann_bp':           generate_ann_bp_raport,
                            'ann_ga':           generate_ann_ga_raport}

        button = raport_generator[raport_type](train_metrics, test_metrics)

        return button


def generate_perceptron_sgd_raport(train_metrics, test_metrics):
    perceptron_sgd_raport.set_metrics(train_metrics, test_metrics)
    if isinstance(train_metrics, list):
        perceptron_sgd_raport.generate_cv_raport('/results_menu')
    else:
        perceptron_sgd_raport.generate_raport('/results_menu')

    return dbc.Col(children=[dbc.Button(id='load-raport-button', children='Wczytaj raport', color='secondary',
                                        size='lg', block=True, href='/models/perceptron_sgd_raport')], width=4)


def generate_perceptron_ga_raport(train_metrics, test_metrics):
    perceptron_ga_raport.generate_raport('/results_menu', train_metrics, test_metrics)

    return dbc.Col(children=[dbc.Button(id='load-raport-button', children='Wczytaj raport', color='secondary',
                                        size='lg', block=True, href='/models/perceptron_ga_raport')], width=4)


def generate_ann_bp_raport(train_metrics, test_metrics):
    ann_bp_raport.set_metrics(train_metrics, test_metrics)
    if isinstance(train_metrics, list):
        ann_bp_raport.generate_ann_bp_cv_raport('/results_menu')
    else:
        ann_bp_raport.generate_ann_bp_split_raport('/results_menu')

    return dbc.Col(children=[dbc.Button(id='load-raport-button', children='Wczytaj raport', color='secondary',
                                        size='lg', block=True, href='/models/ann-bp_raport')], width=4)


def generate_ann_ga_raport(train_metrics, test_metrics):
    ann_ga_raport.generate_raport('/results_menu', train_metrics, test_metrics)

    return dbc.Col(children=[dbc.Button(id='load-raport-button', children='Wczytaj raport', color='secondary',
                                        size='lg', block=True, href='/models/ann_ga_raport')], width=4)