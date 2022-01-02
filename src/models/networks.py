import tensorflow as tf
import tensorflow_probability as tfp




class BRNN(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(BRNN, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(96, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(sequence_length),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class RNN(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(RNN, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128)),
                tf.keras.layers.GRU(64, return_sequences=True),
                tf.keras.layers.GRU(32),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(sequence_length),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class CNN_T(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(CNN_T, self).__init__()
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128, 1)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(7,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 8), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 4), padding='valid'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(sequence_length),
            ]
        )

    def call(self, x):
        out = self.cnn(x)
        return out