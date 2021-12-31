import tensorflow as tf
import tensorflow_probability as tfp




class BRNN(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(BRNN, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class CNN_E2E_Deep(tf.keras.Model):

    def __init__(self, num_labels):
        super(CNN_E2E_Deep, self).__init__()
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(16, 12, 1)),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 3), padding='valid'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Flatten(),
                #tf.keras.layers.Dense(32),
                #tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.cnn(x)
        return out



class CNN_E2E_Big(tf.keras.Model):

    def __init__(self, num_labels):
        super(CNN_E2E_Big, self).__init__()
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(16, 12, 1)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 3), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Flatten(),
                #tf.keras.layers.Dense(32),
                #tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.cnn(x)
        return out



class CNN_Interim(tf.keras.Model):

    def __init__(self, num_labels, latent_dim):
        super(CNN_Interim, self).__init__()
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(64, 48, 1)),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim),
                #tf.keras.layers.BatchNormalization(),
                #tf.keras.layers.Activation(activation='relu'),
                tf.keras.layers.Dense(num_labels, activation='softmax'),
            ]
        )

    def call(self, x):
        out = self.cnn(x)
        return out


def CNN_Interim_Phonemes(num_onset, num_nucleus, latent_dim, lr) -> tf.keras.Model:
    """
    This method builds and returns a Model
    :return:
    """

    x_input = tf.keras.Input(shape=(64, 48, 1), dtype='float32')

    x = tf.keras.layers.InputLayer(input_shape=(64, 48, 1))(x_input)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim)(x)
    
    x_onset = tf.keras.layers.Dense(num_onset, activation='softmax', name="onset")(x)
    x_nucleus = tf.keras.layers.Dense(num_nucleus, activation='softmax', name="nucleus")(x)
    
    optimizer = tf.keras.optimizers.Adam(lr)

    model = tf.keras.Model(inputs=[x_input], outputs=[x_onset, x_nucleus])
    model.compile(optimizer=optimizer, metrics=['accuracy'],
                  loss={"onset":tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        "nucleus":tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)})
    
    return model