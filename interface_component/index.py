import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from interface_component.app import app
from interface_component.menus import main_menu, models_menu, perceptron_sgd_menu, perceptron_ga_menu, ann_bp_menu, \
    ann_ga_menu, data_menu, results_menu
from interface_component.raports import perceptron_sgd_raport, perceptron_ga_raport, ann_bp_raport, ann_ga_raport
from interface_component.help import main_help, perceptron_sgd_help, perceptron_ga_help, ann_bp_help, ann_ga_help, \
    data_help, results_help

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def load_page(pathname):
    dest = {'/':                                main_menu.layout,
            '/models':                          models_menu.layout,
            '/models/perceptron_sgd_menu':      perceptron_sgd_menu.layout,
            '/models/perceptron_sgd_raport':    perceptron_sgd_raport.layout,
            '/models/perceptron_ga_menu':       perceptron_ga_menu.layout,
            '/models/perceptron_ga_raport':     perceptron_ga_raport.layout,
            '/models/ann_bp_menu':              ann_bp_menu.layout,
            '/models/ann-bp_raport':            ann_bp_raport.layout,
            '/models/ann_ga_menu':              ann_ga_menu.layout,
            '/models/ann_ga_raport':            ann_ga_raport.layout,
            '/data_menu':                       data_menu.layout,
            '/results_menu':                    results_menu.layout,
            '/help':                            main_help.layout,
            '/help/perceptron_sgd_help':        perceptron_sgd_help.layout,
            '/help/perceptron_ga_help':         perceptron_ga_help.layout,
            '/help/ann_bp_help':                ann_bp_help.layout,
            '/help/ann_ga_help':                ann_ga_help.layout,
            '/help/data_help':                  data_help.layout,
            '/help/results_help':               results_help.layout,
            }

    if pathname in dest.keys():
        return dest[pathname]
    else:
        return '404'

def run_interface():
    app.run_server(debug=True)


# if __name__ == '__main__':
#     app.run_server(debug=True)