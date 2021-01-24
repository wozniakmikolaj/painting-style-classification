"""Predictor for the CNN Skip connection model"""
# standard library
import os

# internal
from src.utils.config import Config
from src.configs.config import CFG

# external
import tensorflow as tf


class CNNSkipConnectionPredictor:
    """CNNSkipConnectionsPredictor Class"""

    def __init__(self):
        self.config = Config.from_json(CFG)
        self.image_size = self.config.data.image_size

        self.saved_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'default_model')
        self.model = tf.saved_model.load(self.saved_path)
        self.predict = self.model.signatures["serving_default"]

    def preprocess(self, image):
        """Prepares the image to be fed into the model for prediction.

        Args:
            image(np.array): Raw unprocessed image in array format.

        Returns:
            image(tf.float32): Normalized and resized image.
        """
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, (self.image_size, self.image_size))
        return tf.cast(image, tf.float32) / 255.0

    def infer(self, image=None):
        """Classifies the image as one of the painting styles.

        Args:
            image(np.array): image in array format, ready for prediction.

        Returns:
            predicted_classes(array): An array of predictions for every single class.
        """
        image = self.preprocess(image)
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor_image = tf.reshape(tensor_image,
                                  [1, tensor_image.shape[0], tensor_image.shape[1], tensor_image.shape[2]])
        prediction = self.predict(tensor_image)

        predicted_classes = prediction.get('dense_1').numpy().tolist()
        return predicted_classes[0]
