import dash_html_components as html
import dash_bootstrap_components as dbc


layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-main-help',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Pomoc')],
                        style={
                            'position': 'relative',
                            'top': '20%',
                            'textAlign': 'center'
                        })
                ])
            ],
            style={
                'position': 'relative',
                'top': '20%',
                'textAlign': 'center',
                'margin-bottom': '100px'}),

    # ==OPCJE_POMOCY====================================================================================================

    dbc.Row(html.P('Tu główne okno pomocy .'), justify='center'),

    dbc.Row([dbc.Col([dbc.Button('Perceptron SGD pomoc', color='secondary', href='/help/perceptron_sgd_help',
                                 size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row([dbc.Col([dbc.Button('Perceptron GA pomoc', color='secondary', href='/help/perceptron_ga_help', size='lg',
                                 block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row([dbc.Col([dbc.Button('Sieć Neuronowa BP pomoc', color='secondary', href='/help/ann_bp_help', size='lg',
                                 block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row([dbc.Col([dbc.Button('Sieć Neuronowa GA pomoc', color='secondary', href='/help/ann_ga_help', size='lg',
                                 block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row([dbc.Col([dbc.Button('Przeglądanie wyników pomoc', color='secondary', href='/help/results_help', size='lg',
                                 block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row([dbc.Col([dbc.Button('Wczytywanie danych pomoc', color='secondary', href='/help/data_help', size='lg',
                                 block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row([dbc.Col([dbc.Button('Wróć', color='secondary', href='/',
                                         size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),


], fluid=True)