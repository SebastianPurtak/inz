import dash_html_components as html
import dash_bootstrap_components as dbc


layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-perceptron-sgd-help',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Perceptron SGD Pomoc')],
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

    # ==OPCJE_POMOCY====================================================================================================

    dbc.Row(html.P('Tu opcje pomocy.'), justify='center'),

    dbc.Row([dbc.Col([dbc.Button('Wróć', color='secondary', href='/help',
                                         size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px'}),


], fluid=True,
    style={'backgroundColor': '#D3D3D3'})