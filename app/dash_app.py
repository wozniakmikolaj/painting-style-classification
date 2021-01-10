# standard library
import io
import re
import base64

# internal
from src.executors.skip_con_predictor import CNNSkipConnectionPredictor

# external
import numpy as np
from PIL import Image
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly_express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
img_pattern = r'base64,(.*)'

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def serve_layout():
    return html.Div(
        # Basically the whole page
        id="root",
        children=[
            # Main site body
            html.Div(
                id="app-container",
                children=[
                    # Main site title
                    html.Div(
                        id="app-title",
                        children=[
                            html.H2("Image Classification App",
                                    id="title",
                                    style={'text-align': 'center'}),
                            html.Hr()
                        ],
                    ),
                    # Image
                    html.Div(
                        id='output-image-upload'
                    )
                ]
                # style={
                #     'width': '70%',
                #     'left': '175px',
                #     'height': '100%',
                #     'float': 'right',
                #     'position': 'relative',
                # }
            ),
            # Sidebar
            html.Div(
                id="sidebar",
                children=[
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '80%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=True
                    ),
                    html.Br(),
                    html.Button('Classify painting style', id='classify_style'),
                    html.Br(),
                    dcc.Graph(id='probabilities-graph'),
                    html.Br(),
                    html.Div('Predicted class:', id='predicted_class')
                ]
                # style={
                #     'width': '30%',
                #     # 'height': '250px',
                #     'background': 'green',
                #     'position': 'fixed',
                #     'height': '100%',
                # }
            ),
            # hidden div to share data between callbacks in users' browser session
            html.Div(id='intermediate-value', style={'display': 'none'})
        ]
    )


app.layout = serve_layout


def display_image(content):
    return html.Div([
        html.Img(src=content[0]),
        html.Hr()
    ])


# A placeholder callback with img in binary, serving as an intermediate data source
@app.callback(Output('intermediate-value', 'children'),
              Input('upload-image', 'contents'))
def pass_intermediate_value(contents):
    if contents is not None:
        return contents


# Displays uploaded image
@app.callback(Output('output-image-upload', 'children'),
              Input('intermediate-value', 'children'))
def update_img_output(contents):
    if contents is not None:
        children = [
            display_image(contents)
        ]
        return children


def _decode(raw_img_string):
    stripped_img_str = re.search(r'base64,(.*)', str(raw_img_string)).group(1)
    base64_decoded = base64.b64decode(stripped_img_str)
    image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.asarray(image).astype(np.float32)
    return image_np


def classify(image_array):
    predictor = CNNSkipConnectionPredictor()
    predicted_class = predictor.infer(image_array)
    return predicted_class


@app.callback(Output('predicted_class', 'children'),
              Input('intermediate-value', 'children'))
def perform_classification(img_data):
    if img_data is not None:
        decoded_image = _decode(img_data)
        predicted_class = classify(decoded_image)
        success_string = f"The predicted class is: {predicted_class}"
        children = [
             html.H5(success_string)
        ]
        return children

# @app.callback([Output('predicted_class', 'children'),
#               Output('probabilities-graph', 'figure')],
#               [Input('intermediate-value', 'children'),
#               Input('classify_style', '_')])
# def perform_classification(img_data):
#     if img_data is not None:
#         children = [
#              classify_image(img_data)
#         ]
#         figure = px.bar()
#         return children, figure


if __name__ == '__main__':
    app.run_server(debug=True)
