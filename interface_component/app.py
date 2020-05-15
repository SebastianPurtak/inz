import dash
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
# app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True


# interesujące:
# MINTY
# SANDSTONE
# SLATE !
# SOLAR !
# SPACELAB
# SUPERHERO !
# UNITED
# CYBORG !!
# BOOTSTRAP
# COSMO
# DARKLY !!
# FLATLY !
# LUX