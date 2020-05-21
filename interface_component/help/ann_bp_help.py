import dash_html_components as html
import dash_bootstrap_components as dbc


layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-ann-bp-help',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Sieć Neuronowa BP Pomoc')],
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

    dbc.Row(html.P('Wybierz źródło danych - pozwala wybrać zbiór danych, które zostaną użyte w procesie uczenia.'),
            justify='center'),
    dbc.Row(html.P('Wybierz liczbę epok - pozwala określić liczbę epok procesu uczenia.'), justify='center'),
    dbc.Row(html.P('Wybierz wartość współczynnika uczenia - pozwala określić wartość wpsółczynnika learning rate.'),
            justify='center'),
    dbc.Row(html.P('Wprowadź liczbę neuronów w warstwie ukrytej - pozwala określić ile neuronów znajdzie się w warstwie ukrytej sieci neuronowej.'),
            justify='center'),
    dbc.Row(html.P(
        'Wybierz metodę walidacji -  możliwy jest wybór pomiędzy walidacją za pomocą jednego zbioru testowego lub walidacją krzyżową.'),
            justify='center'),
    dbc.Row(html.P(
        'Współczynnik podziału na zbiór treningowy i testowy - określa jaka część zbioru treningowego zostanie wydzielona jako zbiór testowy.'),
            justify='center'),
    dbc.Row(html.P('k zbiorów - pozwala określić na ile zbiorów podzielone zostaną dane w ramach walidacji krzyżowej.'),
            justify='center'),

    dbc.Row([dbc.Col([dbc.Button('Wróć', color='secondary', href='/help',
                                         size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),


], fluid=True)