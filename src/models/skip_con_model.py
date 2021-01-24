"""CNN based model with skip connections"""
# standard library

# internal
from src.executors.skip_con_trainer import CNNSkipConnectionTrainer
from src.models.base_model import BaseModel
from src.data.dataloader import DataLoader
from src.utils.logger import get_logger

# external
import tensorflow as tf

LOG = get_logger('skip_cnn_model')


class CNNSkipConnectionModel(BaseModel):
    """CNNSkipConnectionsModel Class"""

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.model_name = self.config.model.model_name
        self.config = config

        self.dataset = None
        self.dataset_length = None
        self.epochs = self.config.train.epochs

        self.train_dataset = []
        self.validation_dataset = []
        self.test_dataset = []

    def load_data(self):
        """Loads in and preprocesses the data"""
        LOG.info(f'Loading {self.config.data.dataset_name} ...')
        self.train_dataset, self.validation_dataset = DataLoader().etl_load_dataset(self.config)

    def build(self):
        """Builds the tf.keras based model"""
        inputs = tf.keras.Input(shape=self.config.model.input)

        x = tf.keras.layers.Conv2D(64, 5, activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling2D(3)(x)
        x = tf.keras.layers.Conv2D(128, 5, activation="relu")(x)
        block_1_output = tf.keras.layers.MaxPooling2D(3)(x)

        x = tf.keras.layers.Conv2D(128, 5, activation="relu", padding="same")(block_1_output)
        x = tf.keras.layers.Conv2D(128, 5, activation="relu", padding="same")(x)
        block_2_output = tf.keras.layers.concatenate([x, block_1_output])

        x = tf.keras.layers.Conv2D(128, 5, activation="relu", padding="same")(block_2_output)
        x = tf.keras.layers.Conv2D(128, 5, activation="relu", padding="same")(x)
        block_3_output = tf.keras.layers.concatenate([x, block_2_output])

        x = tf.keras.layers.Conv2D(256, 5, activation="relu")(block_3_output)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(self.config.model.output, activation='softmax')(x)

        self.model = tf.keras.Model(inputs, outputs, name=self.config.model.model_name)

    def train(self):
        """Compiles and trains the model"""
        LOG.info('Training started')

        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        val_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        trainer = CNNSkipConnectionTrainer(self.model, self.train_dataset, self.validation_dataset,
                                           loss, optimizer, train_metric, val_metric, self.epochs)
        trainer.train()

    def evaluate(self):
        pass
