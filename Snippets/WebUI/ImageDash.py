import time
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback, ctx, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

DATA_FOLDER = Path(r".")
GLOB_PATTERN = '*.jpg'
UPDATE_TIME_S = 2


class PathData:
    def __init__(self, update_time_s=2000):
        self.image_paths = []
        self.update_time_s = update_time_s
        self.last_update_time = 0
        self.update()

    def update(self):
        if time.time() - self.last_update_time > self.update_time_s:
            self.image_paths = sorted(DATA_FOLDER.glob(GLOB_PATTERN))[::-1]
            self.last_update_time = time.time()

    def filenames(self):
        return [{'column-0': ip.name} for ip in self.image_paths]

    def latest_path(self):
        return str(self.image_paths[0]) if len(self.image_paths) else None


data = PathData(UPDATE_TIME_S)


@callback(Output('tbl', 'data'), Output("latest_but", "n_clicks"), Input('sec_counter', 'n_intervals'), State('data_store', 'data'))
def update_table(sec_counter, data_store):
    data.update()
    if data_store['show-latest'] and data.latest_path() != data_store['image_path']:
        return data.filenames(), 0  # go to the latest image by simulating "Latest" button click
    return data.filenames(), no_update


@callback(Output('log', 'children'), Input('data_store', 'data'))
def set_label_text(data_store):
    if data_store['show-latest']:
        return 'Showing latest image. Select a measurement from the table to inspect.'
    return f'Inspecting {data_store["image_path"]}'


@callback(Output("tbl", "selected_cells"), Output("tbl", "active_cell"), Input("latest_but", "n_clicks"))
def clear_table(n_clicks):
    return [], None


@callback(Output('data_store', 'data'), Input('tbl', 'active_cell'))
def table_selection_changed(index):
    if index is None:
        return {'show-latest': True, 'image_path': data.latest_path()}
    return {'show-latest': False, 'image_path': str(data.image_paths[index['row']])}


@callback(Output('raw-image', 'figure'), Input('data_store', 'data'))
def get_image_fig(data_store=None):  # TODO investigate if all this code is needed

    data_store = data_store or {'image_path': data.latest_path()}
    image_path = data_store['image_path']
    if image_path is None:
        raise PreventUpdate
    image = Image.open(image_path)

    # Create figure
    fig = go.Figure()

    # Constants
    img_width = image.width  # 1600
    img_height = image.height  # 900
    scale_factor = 1

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=image)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    # Disable the autosize on double click because it adds unwanted margins around the image
    # More detail: https://plotly.com/python/configuration-options/
    # fig.show(config={'doubleClick': 'reset', 'displaylogo': False})
    return fig


external_stylesheets = [dbc.themes.BOOTSTRAP]  # dbc.themes.CYBORG
app = Dash(__name__, update_title=None, external_stylesheets=external_stylesheets)
app.title = "Vision Inspection"

try:
    image_view = dcc.Graph(id='raw-image', figure=get_image_fig(), className='card',
                           config={'doubleClick': 'reset', 'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d']})
except PreventUpdate:
    image_view = dcc.Graph(id='raw-image', className='card',
                           config={'doubleClick': 'reset', 'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d']})

log_view = dbc.Label(id='log', className="header-description")
table_view = dash_table.DataTable(data=data.filenames(),
                                  columns=[{'name': 'Acquired Images', 'id': 'column-0'}],
                                  id='tbl', page_size=30, fill_width=True, style_cell={'textAlign': 'center'})
latest_but = dbc.Button("Latest", id='latest_but', n_clicks=0, style={'width': '100%'})

logo = html.Img(src='assets/ProInvent_Logo_Transparent.png', style={'height': 120, 'position':'absolute', 'top': '15px', 'left':'15px'})

app.layout = html.Div(children=[
    html.Div([
        html.H1(children='Vision Inspection', className="header-title"),
        log_view], className="header"),
    dbc.Row([dbc.Col(image_view, width='auto'), dbc.Col([table_view, latest_but], width=2, className='card')], className="wrapper"),
    logo,
    dcc.Interval(id='sec_counter', interval=UPDATE_TIME_S * 1000, n_intervals=0),
    dcc.Store(id='data_store', data={'show-latest': True, 'image_path': None})
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
