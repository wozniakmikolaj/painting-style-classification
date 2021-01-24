"""A simple dashboard for painting style classification, see README.md for more information regarding the
application. """
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
import dash_bootstrap_components as dbc
import plotly_express as px

LABELS_LIST = ['ART NOUVEAU', 'BAROQUE', 'CUBISM', 'EXPRESSIONISM', 'IMPRESSIONISM',
               'REALISM', 'ROMANTICISM', 'SURREALISM', 'SYMBOLISM', 'UKIYO-E']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )


def serve_layout():
    """Prepares the dashboard layout.

    Returns:
         page layout for the dashboard.
    """
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
    """Displays the image once it's uploaded to the website.

    Returns:
        an HTML Div containing the image.
    """
    return html.Div([
        html.Img(src=content[0],
                 style={'display': 'block',
                        'margin-left': 'auto',
                        'margin-right': 'auto'
                        }
                 )
    ])


def _decode(raw_img_string):
    """Decodes an image string from binary.

    Args:
         raw_img_string(str): Raw binary string of an image.

    Returns:
          image_np(np.array): A numpyt array with decoded image.
    """
    stripped_img_str = re.search(r'base64,(.*)', str(raw_img_string)).group(1)
    base64_decoded = base64.b64decode(stripped_img_str)
    image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.asarray(image).astype(np.float32)
    return image_np


def classify(image_array):
    """
    Args:
        image_array(np.array): An image sent in by the user.

    Returns:
        predicted_classes(np.array): An array of predictions for each of the supported classes
    """
    predictor = CNNSkipConnectionPredictor()
    predicted_classes = predictor.infer(image_array)
    return predicted_classes


# convert to return a sorted pandas dataframe instead
def convert_prob_labels(list_of_probabilities, list_of_labels):
    """Converts lists of probabilities and labels so that they are tied together.

    Args:
        list_of_probabilities(list): a list of probabilities sent in from the predictor.
        list_of_labels(list): a list of labels sent in from the predictor.

    Returns:
        probabilities_list(list): A list of the top 5 probabilities of the predictor.
        class_list(list): A list of the art styles corresponding to with the probabilities.
    """
    probabilities_dict = dict(zip(list_of_labels, list_of_probabilities))
    sorted_probabilities_dict = dict(sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True))
    class_list = list(sorted_probabilities_dict.keys())[:5]
    probabilities_list = list(sorted_probabilities_dict.values())[:5]
    return probabilities_list, class_list


def draw_probability_plot(raw_probabilities, class_list=LABELS_LIST):
    """Draws the bar plot of the top 5 predictions with labels.

    Args:
        raw_probabilities(np.array): An array of probabilities for the image sent in by the user.
        class_list(list): A list of all the available art styles.

    Returns:
        figure(px.bar): A plotly express bar plot.
    """
    probabilities, labels = convert_prob_labels(raw_probabilities, class_list)
    figure = px.bar(x=labels, y=probabilities,
                    labels={'x': '', 'y': 'probabilities'})
    return figure


# A placeholder callback with img in binary, serving as an intermediate data source
@app.callback(Output('intermediate-value', 'children'),
              Input('upload-image', 'contents'))
def pass_intermediate_value(contents):
    """Grabs the content sent in by the user and acts as a intermediate callback, so that other callbacks can share the
    data the user sent.

    Args:
        contents: content sent in from the user and the upload-image object.

    Returns:
        contents: content sent in from the user and the upload-image object.
    """
    if contents is not None:
        return contents


# Displays uploaded image
@app.callback(Output('output-image-upload', 'children'),
              Input('intermediate-value', 'children'))
def update_img_output(contents):
    """Grabs the intermediate value (user's content) and displays the image.

    Args:
        contents: content sent in from the user and the upload-image object.

    Returns:
        children: A HTML object that displays the image.
    """
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
    """Grabs the image sent by the user, decodes it, performs art style classification, draws the image and the
    probabilities bar plot.

    Args:
        img_data(str): Raw image data sent in by the user in binary format.

    Returns:
        children: An HTML object that writes out the predicted art style.
        figure: An HTML object that draws the bar plot of the probabilities.
    """
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
