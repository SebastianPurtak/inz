import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from interface_component.app import app
from interface_component.menus import perceptron_sgd_menu, perceptron_ga_menu, main_menu, ann_bp_menu, ann_ga_menu
from interface_component.raports import perceptron_sgd_raport, perceptron_ga_raport, ann_bp_raport, ann_ga_raport
# from apps import main_menu, algorithms_menu, perceptron_sgd_menu, perceptron_ga_menu, nn_bp_menu, nn_ga_menu

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def load_page(pathname):
    # dest = {'/apps/main_menu':              main_menu.layout,
    #         '/apps/algorithms_menu':        algorithms_menu.layout,
    #         '/apps/perceptron_sgd_menu':    perceptron_sgd_menu.layout,
    #         '/apps/perceptron_ga_menu':     perceptron_ga_menu.layout,
    #         '/apps/nn_bp_menu':             nn_bp_menu.layout,
    #         '/apps/nn_ga_menu':             nn_ga_menu.layout}

    dest = {'/': main_menu.layout,
            '/perceptron_sgd_menu':         perceptron_sgd_menu.layout,
            '/perceptron_sgd_raport':       perceptron_sgd_raport.layout,
            '/apps/perceptron_ga_menu':     perceptron_ga_menu.layout,
            '/apps/perceptron_ga_raport':   perceptron_ga_raport.layout,
            '/apps/nn_bp_menu':             ann_bp_menu.layout,
            '/apps/ann-bp_raport':          ann_bp_raport.layout,
            '/apps/nn_ga_menu':             ann_ga_menu.layout,
            '/apps/ann_ga_raport':          ann_ga_raport.layout
            }

    if pathname in dest.keys():
        return dest[pathname]
    else:
        return '404'

def run_interface():
    app.run_server(debug=True)


# if __name__ == '__main__':
#     app.run_server(debug=True)