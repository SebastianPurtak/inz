import dash_html_components as html
import dash_bootstrap_components as dbc


layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-perceptron-ga-help',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Perceptron GA Pomoc')],
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

    dbc.Row(html.P('Wybierz źródło danych - pozwala wybrać zbiór danych, które zostaną użyte w procesie uczenia.'), justify='center'),
    dbc.Row(html.P('Wybierz liczbę pokoleń - określi ilość pokoleń dla algorytmu genetycznego.'), justify='center'),
    dbc.Row(html.P('Wybierz liczebność populacji - określa wielkość populacji początkowej.'), justify='center'),
    dbc.Row(html.P('Wybierz współczynnik selekcji - określa jaka część populacji zostanie wybrana do puli rodziców w każdym pokoleniu.'), justify='center'),
    dbc.Row(html.P('Wybierz rodzaj mutacji - dostępne są dwa rodzaje mutacji: swap_mut - zamienia miejscami dwa geny, random_mut - losuje nową wartość dla poszczególnych genów.'), justify='center'),
    dbc.Row(html.P('Współczynnik mutacji genomu - określa prawdopodobieństwo wystąpienia mutacji u poszczególnych osobników.'), justify='center'),
    dbc.Row(html.P('Współczynnik mutacji genu - odnosi się do mutacji typu random_mut. Parametr ten określa z jakim prawdopodobieństwem zmienione zostaną kolejne geny. '), justify='center'),
    dbc.Row(html.P('Wybierz metodę selekcji - zalecany jest wybór best_selection.'), justify='center'),
    dbc.Row(html.P('Wybierz metodę doboru rodziców - określa sposób w jaki rodzice będą wybierani z puli rodziców dla konkretnej operacji krzyżowania.'), justify='center'),
    dbc.Row(html.P('Wybierz metodę krzyżowania - pozwala na wybór operatora krzyżowania.'), justify='center'),
    dbc.Row(html.P('Wybierz wielkość populacji testowej - określa ilu osobników najlepszych osobników z ostatniego pokolenia zostanie przetestowanych za pomocą zbioru testowego.'), justify='center'),
    dbc.Row(html.P('Współczynnik podziału na zbiór treningowy i testowy - określa jaka część zbioru treningowego zostanie wydzielona jako zbiór testowy.'), justify='center'),
    dbc.Row(html.P('Wybierz minimalną wartość funkcji dopasowania - warunek stopu w postaci określonej wartości funkcji dopasowania. '), justify='center'),

    dbc.Row([dbc.Col([dbc.Button('Wróć', color='secondary', href='/help',
                                         size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),


], fluid=True)