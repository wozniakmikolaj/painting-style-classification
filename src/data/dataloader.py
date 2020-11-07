"""Data Loader"""
# standard library

# internal

# external
import tensorflow as tf


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        return tf.data.Dataset.from_tensor_slices((data_config.path_processed_data,
                                                   data_config.path_processed_labels))
