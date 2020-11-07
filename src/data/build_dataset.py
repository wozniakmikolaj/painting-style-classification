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


def _get_names_and_styles(styles, num_images_per_class=20):
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


def _prepare_img_data(array_of_imgs, array_of_labels, img_width=200, img_height=200):
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
    np.save(data_path, data)
    np.save(labels_path, labels)


def build_dataset():
    new_styles_array = ['Impressionism', 'Realism', 'Romanticism', 'Expressionism',
                        'Art Nouveau (Modern)', 'Baroque', 'Surrealism', 'Symbolism',
                        'Cubism', 'Ukiyo-e']

    images_array, labels_array = _get_names_and_styles(new_styles_array)

    data, labels = _prepare_img_data(images_array, labels_array)

    _save_data_and_labels(data=data, labels=labels)


if __name__ == '__main__':
    build_dataset()
