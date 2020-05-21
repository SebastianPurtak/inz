import dash_html_components as html
import dash_bootstrap_components as dbc


layout = dbc.Container([
    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-main-menu',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Implementacja metod neuroewolucji do celów uczenia sztucznych sieci neuronowych')],
                        style={
                            'position': 'relative',
                            'top': '20%',
                            'margin-bottom': '100px',
                            'margin-top': '20px',
                            'textAlign': 'center'
                        })
                ])
            ]),

    # ==OPCJE_WYBORU====================================================================================================

    dbc.Row([dbc.Col([dbc.Button('Wybierz model', color='secondary', href='/models', size='lg', block=True)],
                     width=4)],
            justify='center', style={'padding': '15px', 'margin-top': '15px', 'margin-bottom': '20px'}),


    dbc.Row([dbc.Col([dbc.Button('Przeglądaj wyniki', color='secondary', href='/results_menu', size='lg', block=True)],
                     width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '20px'}),


    dbc.Row([dbc.Col([dbc.Button('Wczytaj dane treningowe', color='secondary', href='/data_menu', size='lg',
                                 block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '20px'}),


    dbc.Row([dbc.Col([dbc.Button('Pomoc', color='secondary', href='/help', size='lg', block=True)],
                     width=4)],
            justify='center', style={'padding': '15px'}),

],
    fluid=True,)


# 'backgroundColor':   '#D3D3D3'


