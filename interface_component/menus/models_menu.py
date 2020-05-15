import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from interface_component.app import app

help_text = '''
**Perceptron SGD** - algorytm perceptronu sgd\n
**Perceptron GA** - algorytm perceptronu ga\n
**Sieć Neuronowa BP** - algorytm sieci neuronowej bp\n
**Sieć Neuronowa GA** - algorytm sieci neuronowej ga\n

'''

layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-model-menu',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('MENU WYBORU ALGORYTMU')],
                        style={
                            'position': 'relative',
                            'top': '20%',
                            'textAlign': 'center',
                            'margin-bottom': '100px'
                        })
                ])
            ]),

    # ==OPCJE_WYBORU====================================================================================================

    dbc.Row([

        # ==WYBÓR_MODELI================================================================================================
        dbc.Col([

            dbc.Row([dbc.Col([dbc.Button('Perceptron SGD', color='secondary', href='/models/perceptron_sgd_menu',
                                         size='lg', block=True)], width=5)],
            justify='center', style={'padding': '15px', 'margin-bottom': '20px'}),

            dbc.Row([dbc.Col([dbc.Button('Perceptron GA', color='secondary', href='/models/perceptron_ga_menu',
                                         size='lg', block=True)], width=5)],
            justify='center', style={'padding': '15px', 'margin-bottom': '20px'}),

            dbc.Row([dbc.Col([dbc.Button('Sieć Neuronowa BP', color='secondary', href='/models/ann_bp_menu',
                                         size='lg', block=True)], width=5)],
            justify='center', style={'padding': '15px', 'margin-bottom': '20px'}),

            dbc.Row([dbc.Col([dbc.Button('Sieć Neuronowa GA', color='secondary', href='/models/ann_ga_menu',
                                         size='lg', block=True)], width=5)],
            justify='center', style={'padding': '15px', 'margin-bottom': '20px'}),

            dbc.Row([dbc.Col([dbc.Button('Wróć', color='secondary', href='/',
                                         size='lg', block=True)], width=5)],
            justify='center', style={'padding': '15px'}),

        ]),

        # ==OPIS_MODELI================================================================================================
        dbc.Col([

            # html.H3('Tekst pomocy', style={'padding': 40}),

            # html.Div(dcc.Markdown(children=help_text), style={'textAlign': 'left', 'margin': 15})

            html.H5('PERCEPTRON SGD - pojedyńczy neuron uczony za pomocą algorytmu stochastic gradient descent (sgd).',
                    style={'padding': '15px', 'margin-bottom': '20px'}),
            html.H5('PERCEPTRON GA  - pojedyńczy neuron uczony za pomocą algorytmu genetycznego (ga).',
                    style={'padding': '15px', 'margin-bottom': '20px'}),
            html.H5('SIEĆ NEURONOWA BP - Sieć neuronowa uczona za pomocą algorytmu wstecznej propagacji błędu (bp).',
                    style={'padding': '15px', 'margin-bottom': '20px'}),
            html.H5('SIEĆ NEURONOWA GA - Sieć neuronowa uczona za pomocą algorytmu genetycznego (ga).',
                    style={'padding': '15px', 'margin-bottom': '20px'})
        ])

    ], justify='center'),

], fluid=True)