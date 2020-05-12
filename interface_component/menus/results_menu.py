import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output

from interface_component.app import app
from interface_component.utils.db_facade import DBFacade
from interface_component.raports import perceptron_sgd_raport, ann_bp_raport

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
                            'textAlign': 'center'
                        })
                ])
            ],
            style={
                'height': '100px',
                'backgroundColor': '#C0C0C0',
            }),

    # ==OPCJE_PODGLĄDU_WYNIKÓW==========================================================================================

    dbc.Row(html.H4('Wybierz model'), justify='center'),

    dbc.Row(id='raports-list-choice-row', children=[
        dcc.Dropdown(id='raports-list-choice',
                     options=[
                         {'label': value, 'value': value} for value in collections_list
                     ],
                     clearable=False,
                     value=collections_list[0],
                     style={'width': '300px'})
    ],
            justify='center',
            style={'padding': '10px'}),


    dbc.Row(html.H4('Zapisane raporty:'), justify='center'),
    dbc.Row(dbc.Col(id='raports-preview', children=[], width=8), justify='center'),
    dbc.Row(html.H4('Wybierz raport:'), justify='center'),
    dbc.Row(id='raports_list-psgd', children=[dcc.Dropdown(id='raports-choice',
                     options=[{'label': value, 'value': value} for value in {'enpty': 'enpty'}],
                     clearable=False,
                     value='empty',
                     style={'width': '300px'})], justify='center'),

    dbc.Row(id='load_button_row', children=[], justify='center', style={'padding': '10px'}),

    dbc.Row(dbc.Col(children=[dbc.Button(id='delete-raport-button', children='Usuń raport', color='secondary',
                                         size='lg', block=True)], width=4), justify='center', style={'padding': '10px'}),


    # ==KOMUNIKATY_BŁĘDÓW===============================================================================================

    dbc.Row(id='lists-raports-psgd-alert', children=[], justify='center'),
    dbc.Row(id='lists-raports-pga-alert', children=[], justify='center'),
    dbc.Row(id='lists-raports-ann-bp-alert', children=[], justify='center'),
    dbc.Row(id='lists-raports-ann-ga-alert', children=[], justify='center'),
    dbc.Row(id='read-raport-alert', children=[], justify='center'),
    dbc.Row(id='choice-raport-alert', children=[], justify='center'),
    dbc.Row(id='delete-raport-alert', children=[], justify='center'),

    dbc.Row([dbc.Col([dbc.Button('Wróć', color='secondary', href='/',
                                         size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px'}),


], fluid=True,
    style={'backgroundColor': '#D3D3D3'})


@app.callback([Output('raports-preview', 'children'), Output('raports_list-psgd', 'children')],
              [Input('raports-list-choice', 'value')])
def show_collections_list(collection_name):
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
                     style={'padding': '10px'}),

    return table, itable


@app.callback(Output('load_button_row', 'children'), [Input('raports-choice', 'value')])
def choice_raport(value):
    if 'perceptron_sgd' in value:
        test_metrics, train_metrics = db_facade.get_raport_data(value)

        perceptron_sgd_raport.set_metrics(train_metrics, test_metrics)
        if isinstance(train_metrics, list):
            perceptron_sgd_raport.generate_cv_raport('/results_menu')
        else:
            perceptron_sgd_raport.generate_raport('/results_menu')

        return dbc.Col(children=[dbc.Button(id='load-raport-button', children='Wczytaj raport', color='secondary',
                                        size='lg', block=True, href='/models/perceptron_sgd_raport')], width=4)

    elif 'ann_bp' in value:
        test_metrics, train_metrics = db_facade.get_raport_data(value)

        ann_bp_raport.set_metrics(train_metrics, test_metrics)
        if isinstance(train_metrics, list):
            ann_bp_raport.generate_ann_bp_cv_raport('/results_menu')
        else:
            ann_bp_raport.generate_ann_bp_split_raport('/results_menu')

        return dbc.Col(children=[dbc.Button(id='load-raport-button', children='Wczytaj raport', color='secondary',
                                            size='lg', block=True, href='/models/ann-bp_raport')], width=4)


@app.callback(Output('delete-raport-alert', 'children'), [Input('delete-raport-button', 'n_clicks')])
def delete_raport(n_clicks):
    if n_clicks is not None:
        raport_name = db_facade.delete_raport()

        return dbc.Alert(id='delete-info', children=f'Usunięto {raport_name}', color='danger')

