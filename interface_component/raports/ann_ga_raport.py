import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from interface_component.app import app
from interface_component.utils.db_facade import DBFacade
# from interface_component.utils.db_facade import db_facade


layout = {}
train_metrics = None
test_metrics = None
# db_facade = DBFacade()


def generate_raport(back_link, train_data, test_data):
    global layout
    global train_metrics
    global test_metrics

    train_metrics = train_data
    test_metrics = test_data

    raport = dbc.Container([

        # ==NAGŁÓWEK====================================================================================================

        dbc.Row(id='ann-ga-header',
            children=[
                dbc.Col([
                    html.Div([
                        html.H1('Sieć Neuronowa GA Raport')],
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

        dbc.Row(id='train-metrics-ann-ga-header',
                children=html.H4('Metryki procesu uczenia'),
                justify='center',
                style={'padding': '10px',
                       'margin-top': '30px'}),

        # ==FITNESS=====================================================================================================

        dbc.Row(
            dcc.Graph(id='best_fit-ann-ga',
                      figure={'data': [{'x': train_metrics['generation'], 'y': train_metrics['best_fit']}],
                              'layout': {'title': 'Best Fitness',
                                         'xaxis': {'title': 'Pokolenia'},
                                         'yaxis': {'title': 'Best Fitness'},
                                         'plot_bgcolor': '#D3D3D3',
                                         'paper_bgcolor': '#D3D3D3'}}),
        justify='center', style={'padding': '10px'}),

        dbc.Row(
            dcc.Graph(id='mean_fit-ann-ga',
                      figure={'data': [{'x': train_metrics['generation'], 'y': train_metrics['mean_fit']}],
                              'layout': {'title': 'Mean Fitness',
                                         'xaxis': {'title': 'Pokolenia'},
                                         'yaxis': {'title': 'Mean Fitness'},
                                         'plot_bgcolor': '#D3D3D3',
                                         'paper_bgcolor': '#D3D3D3'}}),
            justify='center', style={'padding': '10px'}),

        # ==METRYKI_ZBIORU_TESTOWEGO====================================================================================

        dbc.Row(id='test_metrics-ann-ga-header',
                children=html.H4('Metryki zbioru testowego'),
                justify='center',
                style={'padding': '10px',
                       'margin-bottom': '30px'}),

        dbc.Row(id='test_pop_best_fit-ann_ga-row- table', children=[
            html.Table([
                html.Thead(html.Tr(html.Th('Najlepsze wyniki populacji testowej'))),
                html.Tbody([
                    html.Tr(html.Td(str(result), style={'text-align': 'center'})) for result in test_metrics['val_fit']
                ])
            ])
        ], justify='center'),

        # ==STOPKA======================================================================================================

        dbc.Row(dbc.Col(
            children=[dbc.Button(id='save-raport-ann_ga-button', children='Zapisz raport', color='secondary', size='lg',
                                 block=True)], width=4), justify='center', style={'padding': '10px'}),

        dbc.Row(id='save-raport-ann_ga-alert', children=[], justify='center'),

        dbc.Row([html.Button(id='back_to_config-ann-ga', children=[dcc.Link('Pokaż config', href=back_link)])],
                justify='center',
                style={'padding': '15px'}),
    ],
    fluid=True,
    style={'backgroundColor': '#D3D3D3'})

    layout = raport


@app.callback(Output('save-raport-ann_ga-alert', 'children'), [Input('save-raport-ann_ga-button', 'n_clicks')])
def save_ann_ga_raport(n_clicks):
    if n_clicks is not None:
        db_facade = DBFacade()
        # db_facade.save_raport('ann_ga', {'data': train_metrics}, test_metrics)

        if db_facade.save_raport('ann_ga', {'data': train_metrics}, test_metrics):
            return dbc.Alert(id='save-info', children='Zapisano raport', color='success')
        else:
            return dbc.Alert(id='save-info', children='Nie udało się zapisać raportu', color='danger')