# standard library
import io
import re
import base64

# internal
from src.executors.skip_con_predictor import CNNSkipConnectionPredictor

# external
import numpy as np
import pandas as pd
from PIL import Image
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly_express as px

LABELS_LIST = ['ART NOUVEAU', 'BAROQUE', 'CUBISM', 'EXPRESSIONISM', 'IMPRESSIONISM',
               'REALISM', 'ROMANTICISM', 'SURREALISM', 'SYMBOLISM', 'UKIYO-E']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )


def serve_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([html.H2("Painting style classification", className='text-center'),
                     html.Hr(),
                     html.Br(),
                     html.Br(),
                     html.Div(id='output-image-upload'),
                     html.Br(),
                     html.Br(),
                     html.Hr(),
                     ], width={'size': 8}),
            dbc.Col([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center'
                    },
                    multiple=True
                ),
                html.Br(),
                html.Button(id='classify_style', n_clicks=0,
                            children='Classify painting style',
                            className='btn-block'),
                html.Br(),
                dcc.Graph(id='probabilities-graph'),
                html.Br(),
                html.H6('Predicted class:', id='predicted_class', className='text-center'),
                # hidden div to share data between callbacks in users' browser session
                html.Div(id='intermediate-value', style={'display': 'none'})
            ], width={'size': 4})
        ], justify='center', style={'margin-top': '30px'})
    ], fluid=True)


app.layout = serve_layout


def display_image(content):
    return html.Div([
        html.Img(src=content[0],
                 style={'display': 'block',
                        'margin-left': 'auto',
                        'margin-right': 'auto'
                        }
                 )
    ])


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


# convert to return a sorted pandas dataframe instead
def convert_prob_labels(list_of_probabilities, list_of_labels):
    probabilities_dict = dict(zip(list_of_labels, list_of_probabilities))
    sorted_probabilities_dict = dict(sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True))
    class_list = list(sorted_probabilities_dict.keys())[:5]
    probabilities_list = list(sorted_probabilities_dict.values())[:5]
    return probabilities_list, class_list


def draw_probability_plot(raw_probabilities, class_list=LABELS_LIST):
    probabilities, labels = convert_prob_labels(raw_probabilities, class_list)
    figure = px.bar(x=labels, y=probabilities,
                    labels={'x': '', 'y': 'probabilities'})
    return figure


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


# Draws probability graph and
@app.callback([Output('predicted_class', 'children'),
               Output('probabilities-graph', 'figure')],
              Input('intermediate-value', 'children'))
def perform_classification(img_data):
    if img_data is not None:
        decoded_image = _decode(img_data)
        class_probabilities = classify(decoded_image)
        predicted_class = np.argmax(class_probabilities)
        success_string = f'Painting style : {LABELS_LIST[predicted_class]}'
        children = [
             html.H5(success_string)
        ]
        figure = draw_probability_plot(class_probabilities)
        return children, figure


if __name__ == '__main__':
    app.run_server(debug=True)
