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
    def etl_load_dataset(data_config):
        """Performs the whole data pipeline, from load and processing to returning split data ready for analysis.

        Args:
            data_config(config): Config class passed with the model.

        Returns:
            train_dataset(tf.data.Dataset): transformed train dataset
            validation_dataset(tf.data.Dataset): transformed validation dataset
        """
        dataset = DataLoader()._load_data(data_config)
        train_dataset, validation_dataset = DataLoader().create_datasets(dataset, data_config)

        return train_dataset, validation_dataset

    @staticmethod
    def create_datasets(dataset, data_config):
        """Splits the dataset, shuffles, batches and returns the data ready for analysis.

        Args:
            dataset(tf.data.Dataset): Processed dataset.
            data_config(config): Config class passed with the model.

        Returns:
            train_dataset(tf.data.Dataset): transformed train dataset
            validation_dataset(tf.data.Dataset): transformed validation dataset
        """
        dataset_length = tf.data.experimental.cardinality(dataset).numpy()

        train_take_size = int(dataset_length * data_config.data.train_split)
        validation_take_size = int(dataset_length * data_config.data.validation_split)
        test_take_size = int(dataset_length * data_config.data.test_split)

        train_dataset = dataset.take(train_take_size)
        validation_dataset = dataset.skip(train_take_size)

        train_dataset = train_dataset.cache().shuffle(data_config.train.buffer_size).batch(data_config.train.batch_size,
                                                                                           drop_remainder=True).repeat()

        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.batch(data_config.train.batch_size)

        return train_dataset, validation_dataset

    @staticmethod
    def _load_data(data_config):
        """"Loads in the raw data, processes it, encodes string labels to numeric.

        Args:
            data_config(config): Config class passed with the model.

        Returns:
            dataset(tf.data.Dataset): A processed dataset.
        """
        data = np.load(data_config.paths.path_processed_data)
        labels = np.load(data_config.paths.path_processed_labels)

        data, labels = DataLoader()._prepare_data(data, labels, data_config)

        le = LabelEncoder()
        labels = le.fit_transform(labels).astype('int8')

        return tf.data.Dataset.from_tensor_slices((data, labels))

    @staticmethod
    def _prepare_data(data, labels, data_config):
        """Shuffles, reshapes (adding an extra dimension) and normalizes data and label arrays.

        Args:
            data(np.array): Raw data array.
            labels(np.array): Raw labels array.
            data_config(config): Config class passed with the model.

        Returns:
            data(np.array): transformed data array
            labels(np.array): transformed labels array
        """
        data = np.random.RandomState(7).permutation(data)
        labels = np.random.RandomState(7).permutation(labels)

        data = data.reshape(-1, data_config.data.image_size, data_config.data.image_size,
                            data_config.data.num_channels).astype('float32')
        data = data / 255.

        return data, labels
