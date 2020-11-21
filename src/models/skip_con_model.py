"""CNN based model with skip connections"""
# standard library
import os

# internal
from src.models.base_model import BaseModel
from src.data.dataloader import DataLoader

# external
import tensorflow as tf


class CNNSkipConnectionModel(BaseModel):
    """CNNSkipConnectionsModel Class"""

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.model_name = self.config.model.model_name
        self.output_channels = self.config.model.output

        self.dataset = None
        self.batch_size = self.config.train.batch_size
        self.buffer_size = self.config.train.buffer_size
        self.epochs = self.config.train.epochs
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = 0

        self.image_size = self.config.data.image_size
        self.num_channels = self.config.data.num_channels
        self.train_dataset = []
        self.validation_dataset = []
        self.test_dataset = []

    def load_data(self):
        """Loads in and preprocesses the data"""
        self.dataset = DataLoader().load_data(self.config)
        self._preprocess_data()

    def _preprocess_data(self):
        """ Splits into training, validation and test; sets training parameters"""
        train_take_size = int(10998*self.config.data.train_split)

        self.train_dataset = self.dataset.take(train_take_size)
        self.validation_dataset = self.dataset.skip(train_take_size)

        self._set_training_parameters()

        self.train_dataset = self.train_dataset.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()
        self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.validation_dataset = self.validation_dataset.batch(self.batch_size)

    def _split_dataset(self):
        pass

    def _set_training_parameters(self):
        """Sets training parameters"""

        self.train_length = tf.data.experimental.cardinality(self.train_dataset).numpy()
        self.steps_per_epoch = self.train_length // self.batch_size
        self.validation_steps = tf.data.experimental.cardinality(self.validation_dataset).numpy() // self.batch_size
        # print(f"train length:{self.train_length}, steps per epoch:{self.steps_per_epoch},
        # validation steps:{self.validation_steps}")

    def build(self):
        """Builds the Keras Functional API based model"""
        inputs = tf.keras.Input(shape=self.config.model.input, name=self.model_name)

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

        self.model = tf.keras.Model(inputs, outputs, name="paintings_resnet")

    def _draw_model_plot(self):
        plot_file = os.path.join(self.config.paths.path_model_plot, self.model_name)

        tf.keras.utils.plot_model(self.model, to_file=plot_file, show_shapes=True)

    def train(self):
        """Compiles and trains the model"""
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.train.learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                           metrics=self.config.train.metrics)

        # self._draw_model_plot()

        model_history = self.model.fit(self.train_dataset,
                                       epochs=self.epochs,
                                       steps_per_epoch=self.steps_per_epoch,
                                       # validation_steps=self.validation_steps,
                                       validation_data=self.validation_dataset)

        return model_history.history['loss'], model_history.history['val_loss']

    def evaluate(self):
        print('evaluated the model')
        pass
