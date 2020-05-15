import dash_html_components as html
import dash_bootstrap_components as dbc


layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-ann-ga-help',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Sieć Neuronowa Pomoc')],
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

    dbc.Row(html.P('Tu opcje pomocy.'), justify='center'),

    dbc.Row([dbc.Col([dbc.Button('Wróć', color='secondary', href='/help',
                                         size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),


], fluid=True)