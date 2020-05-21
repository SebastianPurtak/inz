import dash_html_components as html
import dash_bootstrap_components as dbc


layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-data-help',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('WCZYTYWANIE DANYCH POMOC')],
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

    dbc.Row(html.P('Przeciągnij lub wskaż plik - pozwala wskazać plik z nowymi danymi. Po wybraniu plików pojawi się podgląd danych.'), justify='center'),
    dbc.Row(html.P('Normalizacja, Label Encoding, Mieszaj - opcję przetwarzania danych. Po wybraniu którejś z nich pojawi się podgląd przetworzonych danych.'), justify='center'),
    dbc.Row(html.P('Podaj nazwę pliku - pole pozwalające nadać nazwę nowemu plikowi z danymi. Jego wypełnienie jest niezbędne do zapisania danych'), justify='center'),
    dbc.Row(html.P('Aby zapisać przetworzone dane, należy wybrać opcję Zapisz. '), justify='center'),

    dbc.Row([dbc.Col([dbc.Button('Wróć', color='secondary', href='/help',
                                         size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),


], fluid=True)