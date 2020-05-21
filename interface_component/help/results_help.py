import dash_html_components as html
import dash_bootstrap_components as dbc


layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-results-help',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('PRZEGLĄDANIE WYNIKÓW POMOC')],
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

    dbc.Row(html.P('Wybierz model - pozwala wybrać model dla którego zostanie wybrana kolekcja danych.'), justify='center'),
    dbc.Row(html.P('Wybierz raport - pozwala wybrać konkretny raport ze wskazanej kolekcji.'), justify='center'),
    dbc.Row(html.P('Po wybraniu kolekcji i raportu pojawi się przycisk Wczytaj raport, który pozwoli na jego wygenerowanie.'), justify='center'),
    dbc.Row(html.P('Przeciągnij lub wskaż plik - pozwala wskazać zapisany raport. Po wybraniu pliku pojawi się przycisk pozwalający wygenerować raport.'), justify='center'),

    dbc.Row([dbc.Col([dbc.Button('Wróć', color='secondary', href='/help',
                                         size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px'}),


], fluid=True)