# standard library
import os

# internal
from src.utils.logger import get_logger

# external
import tensorflow as tf

LOG = get_logger('trainer')


class CNNSkipConnectionTrainer:

    def __init__(self, model, train_data_input, val_data_input, loss_fn,
                 optimizer, train_metric, val_metric, epochs):
        self.model = model
        self.train_data_input = train_data_input
        self.val_data_input = val_data_input
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_acc_metric = train_metric
        self.val_acc_metric = val_metric
        self.epochs = epochs

        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        self.checkpoint_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'tf_checkpoints')
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=3)

        self.train_log_dir = os.path.join(os.path.dirname(os.getcwd()), 'models', 'logs', 'gradient_tape')
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        self.model_save_path = os.path.join(os.path.dirname(os.getcwd()), 'models')

    @tf.function
    def train_step(self, batch):
        trainable_variables = self.model.trainable_variables
        inputs, labels = batch
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            step_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(step_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        self.train_acc_metric.update_state(labels, predictions)

        return step_loss, predictions

    @tf.function
    def validation_step(self, batch):
        inputs, labels = batch
        validations = self.model(inputs, training=False)
        self.val_acc_metric.update_state(labels, validations)

    def train(self):
        for epoch in range(self.epochs):
            LOG.info(f'Start epoch {epoch}')

            step_loss = 0
            for step, training_batch in enumerate(self.train_data_input):
                step_loss, predictions = self.train_step(training_batch)
                LOG.info(f"Loss at step {step}: {step_loss:.2f}; Samples seen so far: {((step + 1) * 64)}")
                if step == 10:
                    break

            train_acc = self.train_acc_metric.result()
            LOG.info(f"Training acc over epoch: {float(train_acc):.4f}")

            save_path = self.checkpoint_manager.save()
            LOG.info(f"Saved checkpoint: {save_path}")

            self._write_summary(step_loss, epoch)

            self.train_acc_metric.reset_states()

            for validation_batch in self.val_data_input:
                self.validation_step(validation_batch)

            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_states()
            print(f"Validation acc: {float(val_acc):.4f}")

        save_path = os.path.join(self.model_save_path, 'skip_con', '1')
        tf.saved_model.save(self.model, save_path)
        LOG.info(f"Model {save_path} saved.")

    def _write_summary(self, loss, epoch):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=epoch)
            tf.summary.scalar('accuracy', self.train_acc_metric.result(), step=epoch)
            # tensorboard --logdir logs/gradient_tape

    def print_out_paths(self):
        print(f'Saved checkpoint path: {self.checkpoint_manager.save()}')
        print(f'Saved model path: {self.model_save_path}')
        print(f'Log path: {self.train_log_dir}')
