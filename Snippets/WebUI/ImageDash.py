from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

DATA_FOLDER = Path(r".")


class PathData:
    def __init__(self):
        self.image_paths = []
        self.update()

    def update(self):
        self.image_paths = sorted(DATA_FOLDER.glob("*.png"))

    def filenames(self):
        return [{'column-0': ip.name} for ip in self.image_paths]


data = PathData()


@callback(Output('tbl', 'data'), Input('sec_counter', 'n_intervals'))
def update_table(sec_counter=None):
    data.update()
    return data.filenames()


@callback(Output('log', 'children'), Input('tbl', 'active_cell'))
def set_label_text(index):
    if index is None:
        return 'Select a measurement from the table to inspect.'
    return f'Inspecting {data.image_paths[index["row"]]}'


@callback(Output('raw-image', 'figure'), Input('tbl', 'active_cell'))
def get_image_fig(index):  # TODO investigate if all this code is needed
    if len(data.image_paths) == 0 or index is None:
        raise PreventUpdate()

    if isinstance(index, dict):
        index = index['row']
    image = Image.open(data.image_paths[index])

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
    # fig.show(config={'doubleClick': 'reset'})
    return fig


external_stylesheets = [dbc.themes.BOOTSTRAP]  # dbc.themes.CYBORG
app = Dash(__name__, external_stylesheets=external_stylesheets)

log_view = dbc.Label(id='log')
image_view = dcc.Graph(id='raw-image')
table_view = dash_table.DataTable(data=data.filenames(),
                                  columns=[{'name': 'Acquired Images', 'id': 'column-0'}],
                                  id='tbl', page_size=40, fill_width=False, style_cell={'textAlign': 'left'})

app.layout = html.Div(children=[
    html.H1(children='Vision Inspection'),
    log_view,
    html.Hr(),
    dbc.Row([dbc.Col(image_view, width='auto'), dbc.Col(table_view)]),
    dcc.Interval(id='sec_counter', interval=1 * 1000, n_intervals=0)
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
