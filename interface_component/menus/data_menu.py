import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_table

from interface_component.app import app

from interface_component.utils.data_management import DataManagement

data_manager = DataManagement()

filename = ''


layout = dbc.Container([

    # ==NAGŁÓWEK========================================================================================================

    dbc.Row(id='header-data-menu',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Zarządzanie danymi')],
                        style={
                            'position': 'relative',
                            'top': '20%',
                            'textAlign': 'center',
                            'margin-bottom': '100px'
                        })
                ])
            ]),

    # ==WCZYTYWANIE_PLIKU===============================================================================================

    dbc.Row(id='upload-file-header',
                children=html.H4('Wczytaj nowy plik z danymi'),
                justify='center',
                style={'padding': '10px',
                       'margin-top': '30px',
                       'margin-bottom': '10px'}),

    dbc.Row(dbc.Col([dcc.Upload(id='upload-data', children=html.Div(['Przeciągnij albo ', html.A('wskaż plik')]),
                                style={'width': '100%',
                                       'height': '60px',
                                       'lineHeight': '60px',
                                       'borderWidth': '1px',
                                       'borderStyle': 'dashed',
                                       'borderRadius': '5px',
                                       'textAlign': 'center',
                                       'margin': '10px'})], width=4), justify='center',
            style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row(html.H4('Wprowadzone dane:'), justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row(dbc.Col(id='upload-data-preview', children=[], width=8), justify='center',
            style={'padding': '15px', 'margin-bottom': '10px'}),

    # ==OPCJE_PRZETWARZANIA_DANYCH======================================================================================

    dbc.Row([
        dbc.Col([dbc.Button(id='normalization-button', children='Normalizacja', color='secondary', size='lg',
                                 block=True)], width=2),
        dbc.Col([dbc.Button(id='label-encoding-button', children='Label Encoding', color='secondary', size='lg',
                                 block=True)], width=2),
        dbc.Col([dbc.Button(id='label-shuffle-button', children='Mieszaj', color='secondary', size='lg',
                            block=True)], width=2)], justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row(html.H4('Przetworzone dane:'), justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row(dbc.Col(id='processed-data-preview', children=[], width=8), justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    # ==PODGLĄD_DOSTĘPNYCH_DANYCH=======================================================================================

    dbc.Row(html.H4('Dostępne zbiory danych:'), justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row(id='datasets_preview', children=[], justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),

    dbc.Row(dbc.Col([dbc.Button(id='refresh-button', children='Odświerz', color='secondary', size='lg',
                                block=True)], width=4), justify='center',
            style={'padding': '10px', 'margin-bottom': '10px'}),

    # ==KOMUNIKATY_BŁĘDÓW===============================================================================================

    dbc.Row(dbc.Col([dbc.Input(id='filename_input', placeholder='Podaj nazwę pliku', type='text')], width=4),
            style={'padding': '10px', 'margin-bottom': '10px'}, justify='center'),

    dbc.Row(dbc.Col([dbc.Button(id='save-button', children='Zapisz', color='secondary', size='lg',
                                 block=True)], width=4), justify='center',
            style={'padding': '10px', 'margin-bottom': '10px'}),

    dbc.Row(html.P('Komunikaty'), justify='center', style={'padding': '10px', 'margin-bottom': '10px'}),

    dbc.Row(id='data-read-error', children=[], justify='center'),
    dbc.Row(id='data-preprocess-error', children=[], justify='center'),
    dbc.Row(id='data-normalization-error', children=[], justify='center'),
    dbc.Row(id='data-label_encoding-error', children=[], justify='center'),
    dbc.Row(id='data-shuffle_encoding-error', children=[], justify='center'),
    dbc.Row(id='save-error', children=[], justify='center'),
    dbc.Row(id='filename-error', children=[], justify='center'),



    dbc.Row([dbc.Col([dbc.Button(id='Back', children='Wróć', color='secondary', href='/',
                                         size='lg', block=True)], width=4)],
            justify='center', style={'padding': '15px', 'margin-bottom': '10px'}),


], fluid=True)


@app.callback([Output('upload-data-preview', 'children'), Output('data-read-error', 'children')],
              [Input('upload-data', 'contents')],
               [State('upload-data', 'filename'), State('upload-data', 'last_modified')])
def upload_file(data, filename, last_modified):
    if data is not None:
        status = data_manager.read_data(data)

        if status:
            data_preview = data_manager.get_data()

            data_table = dash_table.DataTable(columns=[{'name': i, 'id': i} for i in data_preview.columns],
                                              data=data_preview.to_dict('records'),
                                              style_data={'whiteSpace': 'normal',
                                                          'height': 'auto'},
                                              style_table={'overflowY': 'scroll',
                                                           'overflowX': 'scroll',
                                                           'maxHeight': '500px'},
                                              style_cell={'backgroundColor': 'rgb(100, 100, 100)',
                                                          'color': 'white',
                                                          'width': '100px'},
                                              style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                                              style_data_conditional=[{'if': {'column_id': 'Answer'},
                                                                       'backgroundColor': 'rgb(30, 30, 30)',
                                                                       'color': 'white'}])

            return data_table, None

        else:
            return None, dbc.Alert(id='data-read_alert',
                                   children='Wystąpił problem z wczytaniem danych.',
                                   color='danger')

    else:
        return None, None


@app.callback(Output('data-normalization-error', 'children'),
              [Input('normalization-button', 'n_clicks')])
def normalization_data(normalization_click):
    if normalization_click is not None:
        data_manager.data_normalization()

        return None


@app.callback(Output('data-label_encoding-error', 'children'),
              [Input('label-encoding-button', 'n_clicks')])
def label_encoding_data(normalization_click):
    if normalization_click is not None:
        data_manager.label_encoding()

        return None


@app.callback(Output('data-shuffle_encoding-error', 'children'),
              [Input('label-shuffle-button', 'n_clicks')])
def shuffle_data(shuffle_click):
    if shuffle_click is not None:
        data_manager.shuffle_data()

        return None


@app.callback(Output('processed-data-preview', 'children'),
              [Input('normalization-button', 'n_clicks'), Input('label-encoding-button', 'n_clicks'),
               Input('label-shuffle-button', 'n_clicks')])
def preprocess_data(normalization_click, label_encoding_click, shuffle_clicks):
    if normalization_click is not None or label_encoding_click is not None or shuffle_clicks is not None:
        data_preview = data_manager.get_data()

        data_table = dash_table.DataTable(columns=[{'name': i, 'id': i} for i in data_preview.columns],
                                          data=data_preview.to_dict('records'),
                                          style_data={'whiteSpace': 'normal',
                                                      'height': 'auto'},
                                          style_table={'overflowY': 'scroll',
                                                       'overflowX': 'scroll',
                                                       'maxHeight': '500px'},
                                          style_cell={'backgroundColor': 'rgb(100, 100, 100)',
                                                      'color': 'white',
                                                      'width': '100px'},
                                          style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                                          style_data_conditional=[{'if': {'column_id': 'Answer'},
                                                                   'backgroundColor': 'rgb(30, 30, 30)',
                                                                   'color': 'white'}])

        return data_table


@app.callback(Output('datasets_preview', 'children'), [Input('refresh-button', 'n_clicks')])
def datasets_preview(n_clicks):
    datasets = data_manager.get_datasets_list()
    datasets.remove('.gitkeep')

    table = dbc.Row(id='datasets-list-table', children=[
        html.Table([
            html.Tbody([
                html.Tr(html.Td(str(result), style={'text-align': 'center'})) for result in datasets
            ])
        ])
    ], justify='center'),

    return table


@app.callback(Output('filename-error', 'children'), [Input('filename_input', 'value')])
def set_filename(name):
    global filename
    filename = name


@app.callback(Output('save-error', 'children'), [Input('save-button', 'n_clicks')])
def save_file(save_click):
    global filename
    if save_click is not None and filename is None:
        return dbc.Alert(id='save-data_alert', children='Wprowadź nazwę pliku.', color='danger')

    if save_click is not None and filename is not None:
        data_manager.save_data(filename)

        return dbc.Alert(id='save-data_success', children='Zapisano plik.', color='success')



