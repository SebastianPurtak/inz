import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

layout = {}

def generate_raport(train_metrics, test_metrics):
    global layout

    raport = dbc.Container([

        # ==NAGŁÓWEK====================================================================================================

        dbc.Row(id='pga-header',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Perceptron GA Raport')],
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

        dbc.Row(id='train-metrics-pga-header',
                children=html.H4('Metryki procesu uczenia'),
                justify='center',
                style={'padding': '10px',
                       'margin-top': '30px'}),

        # ==FITNESS=====================================================================================================

        dbc.Row(
            dcc.Graph(id='best_fit-pga',
                      figure={'data': [{'x': train_metrics['generation'], 'y': train_metrics['best_fit']}],
                              'layout': {'title': 'Best Fitness',
                                         'xaxis': {'title': 'Pokolenia'},
                                         'yaxis': {'title': 'Best Fitness'},
                                         'plot_bgcolor': '#D3D3D3',
                                         'paper_bgcolor': '#D3D3D3'}}),
        justify='center', style={'padding': '10px'}),

        dbc.Row(
            dcc.Graph(id='mean_fit-pga',
                      figure={'data': [{'x': train_metrics['generation'], 'y': train_metrics['mean_fit']}],
                              'layout': {'title': 'Mean Fitness',
                                         'xaxis': {'title': 'Pokolenia'},
                                         'yaxis': {'title': 'Mean Fitness'},
                                         'plot_bgcolor': '#D3D3D3',
                                         'paper_bgcolor': '#D3D3D3'}}),
            justify='center', style={'padding': '10px'}),

        # ==METRYKI_ZBIORU_TESTOWEGO====================================================================================

        dbc.Row(id='test_metrics-pga-header',
                children=html.H4('Metryki zbioru testowego'),
                justify='center',
                style={'padding': '10px',
                       'margin-bottom': '30px'}),

        dbc.Row(id='test_pop_best_fit-pga-row- table', children=[
            html.Table([
                html.Thead(html.Tr(html.Th('Najlepsze wyniki populacji testowej'))),
                html.Tbody([
                    html.Tr(html.Td(str(result), style={'text-align': 'center'})) for result in test_metrics['val_fit']
                ])
            ])
        ], justify='center'),

        # ==STOPKA======================================================================================================

        dbc.Row([html.Button(id='back_to_config-pga', children=[dcc.Link('Pokaż config',
                                                                         href='/models/perceptron_ga_menu')])],
                justify='center',
                style={'padding': '15px'}),
    ],
    fluid=True,
    style={'backgroundColor': '#D3D3D3'})

    layout = raport