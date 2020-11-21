"""Data Loader"""
# standard library

# internal

# external
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        data = np.load(data_config.paths.path_processed_data)
        labels = np.load(data_config.paths.path_processed_labels)

        print(data.shape, data.nbytes)

        data = data.reshape(-1, data_config.data.image_size, data_config.data.image_size,
                            data_config.data.num_channels).astype('float32')
        data = data / 255.

        print(data.shape, data.nbytes)

        le = LabelEncoder()
        labels = le.fit_transform(labels)

        return tf.data.Dataset.from_tensor_slices((data, labels))


