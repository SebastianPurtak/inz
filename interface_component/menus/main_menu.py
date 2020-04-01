import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from interface_component.app import app

colors = {
    'background': '#D3D3D3',
    'text': '#7FDBFF'
}

left_table_color = {
    'background': '#BDB76B'
}

help_text = '''
**Perceptron SGD** - algorytm perceptronu sgd\n
**Perceptron GA** - algorytm perceptronu ga\n
**Sieć Neuronowa BP** - algorytm sieci neuronowej bp\n
**Sieć Neuronowa GA** - algorytm sieci neuronowej ga\n

'''

layout = html.Div([

    html.Div([
        html.H3('Menu wyboru algorytmu', style={'padding': 40, 'backgroundColor': '#C0C0C0'})
    ]),

    html.Div([
        html.H3('Przyciski', style={'padding': 40}),

        html.Div([
            html.Button(id='chose-perceptron-sgd', children=[dcc.Link('Perceptron SGD', href='/perceptron_sgd_menu')])

        ], style={'padding': 10}),

        html.Div([
            html.Button(id='chose-perceptron-ga', children=[dcc.Link('Perceptron GA', href='/apps/perceptron_ga_menu')])

        ], style={'padding': 10}),

        html.Div([
            html.Button(id='chose-nn-bp', children=[dcc.Link('Sieć Neuronowa BP', href='/apps/nn_bp_menu')])

        ], style={'padding': 10}),

        html.Div([
            html.Button(id='chose-ga', children=[dcc.Link('Sieć Neuronowa GA', href='/apps/nn_ga_menu')])

        ], style={'padding': 10}),

    ], style={
        'width': '49%',
        'display': 'inline-block',
        'backgroundColor': '#D3D3D3',
        'height': '65%'
    }),

    html.Div([
        html.H3('Tekst pomocy', style={'padding': 40}),

        html.Div(dcc.Markdown(children=help_text), style={'textAlign': 'left', 'margin': 15})

    ], style={
        'width': '49%',
        'display': 'inline-block',
        'backgroundColor': '#DCDCDC',
        'height': '65%',
        'vertical-align': 'middle'
    }),

    # html.Div([
    #     html.Button(id='back', children=[dcc.Link('Wróć', href='/apps/main_menu')])],
    #     style={'padding': 10}),


], style={
    'textAlign': 'center',
    'backgroundColor': colors['background'],
    'height': '100vh',
    'verticalAlign': 'middle',
    'top': '0px',
    'left': '0px'
})