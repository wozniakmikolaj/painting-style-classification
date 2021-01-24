"""Script responsible for loading in the images from raw jpg files, readying them for analysis and preparing its
labels """
# standard library
import os

# internal
from src.configs.config import CFG
from src.utils.config import Config

# external
import numpy as np
import pandas as pd
import cv2

config = Config.from_json(CFG)

new_styles_array = ['Impressionism', 'Realism', 'Romanticism', 'Expressionism',
                    'Art Nouveau (Modern)', 'Baroque', 'Surrealism', 'Symbolism',
                    'Cubism', 'Ukiyo-e']


def _get_names_and_styles(styles, num_images_per_class=config.data.images_per_class):
    """Populates the array of images and labels for analysis, from a data description file.

    Args:
        styles(np.array): A list of art styles to iterate over.
        num_images_per_class(int): A number of images in each of the classes, from config file.

    Returns:
        images_array(np.array): A numpy array of images.
        labels_array(np.array): A numpy array of labels.
    """
    painting_data = pd.read_csv(config.paths.path_raw_csv_file)
    images_array = []
    labels_array = []

    for style in styles:
        concatenate_images = painting_data[painting_data['style'] == style].head(num_images_per_class)[
            'filename'].values
        concatenate_labels = painting_data[painting_data['style'] == style].head(num_images_per_class)['style'].values
        images_array = np.append(images_array, concatenate_images)
        labels_array = np.append(labels_array, concatenate_labels)

    return images_array, labels_array


def _prepare_img_data(array_of_imgs, array_of_labels, img_width=config.data.image_size,
                      img_height=config.data.image_size):
    """Populates the array of images and labels for analysis, from a data description file.

    Args:
        array_of_imgs(np.array): A list of art styles to iterate over.
        array_of_labels(np.array): A list of labels to iterate over.
        img_width(int): Width of an image, from config file.
        img_height(int): Height of an image, from config file.

    Returns:
        X(np.array): A numpy array of all the images, after processing.
        y(np.array): A numpy array of all the labels, after processing.
    """
    X = []
    y = []

    for image, label in zip(array_of_imgs, array_of_labels):
        current_file = os.path.join(config.paths.path_raw, image)

        try:
            X.append(np.array(cv2.resize((cv2.imread(current_file, cv2.IMREAD_GRAYSCALE)),
                                         (img_width, img_height))))
            y.append(label.upper())
        except Exception as e:
            print(f"Corrupted image: {image}, skipping the image")
            print(e)

    return X, y


def _save_data_and_labels(data, labels,
                          data_path=config.paths.path_processed_data,
                          labels_path=config.paths.path_processed_labels):
    """Saves the data and its labels as a .npz array.

    Args:
        data(np.array): an array of images ready for analysis.
        labels(np.array) an array of labels ready for analysis.
        data_path: location of the data, from config file.
        labels_path: location of the labels, from config file.

    """
    np.save(data_path, data)
    np.save(labels_path, labels)


def build_dataset(styles_array):
    """Builds the dataset from raw image files and a csv description file, saves the data in .npy format.

    Args:
        styles_array(np.array): an array of art styles we want to analyze.
    """
    images_array, labels_array = _get_names_and_styles(styles_array)

    data, labels = _prepare_img_data(images_array, labels_array)

    _save_data_and_labels(data=data, labels=labels)


if __name__ == '__main__':
    build_dataset()

