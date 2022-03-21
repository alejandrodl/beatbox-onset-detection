import pdb
import tensorflow as tf
import tensorflow_probability as tfp




class BRNN_1(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(BRNN_1, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(sequence_length),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class BRNN_2(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(BRNN_2, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(sequence_length),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class BRNN_3(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(BRNN_3, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(sequence_length),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



'''class RNN_1(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(RNN_1, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128)),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out'''



class RNN_1(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(RNN_1, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class RNN_2(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(RNN_2, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128)),
                tf.keras.layers.LSTM(64, return_sequences=True),

                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class RNN_3(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(RNN_3, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128)),
                tf.keras.layers.LSTM(96, return_sequences=True),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class RNN_Stateful_1(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(RNN_Stateful_1, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(batch_input_shape=(1,1,128)),
                tf.keras.layers.LSTM(16, return_sequences=False, stateful=True),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class RNN_Stateful_2(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(RNN_Stateful_2, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(batch_input_shape=(1,1,128)),
                tf.keras.layers.LSTM(32, return_sequences=False, stateful=True),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class RNN_Stateful_3(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(RNN_Stateful_3, self).__init__()
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(batch_input_shape=(1,1,128)),
                tf.keras.layers.LSTM(64, return_sequences=False, stateful=True),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):
        out = self.rnn(x)
        return out



class CNN_T_1(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(CNN_T_1, self).__init__()
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



class CNN_T_2(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(CNN_T_2, self).__init__()
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128, 1)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(7,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(1, 4), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 4), padding='valid'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
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



class CNN_T_3(tf.keras.Model):

    def __init__(self, sequence_length, dropout):
        super(CNN_T_3, self).__init__()
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, 128, 1)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(7,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='valid'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 4), padding='valid'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation=None, padding='same'),
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



class CRNN_1(tf.keras.Model):

    def __init__(self, kernel_length, num_filters, sequence_length, dropout):
        super(CRNN_1, self).__init__()
        self.sequence_length = sequence_length
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 1)),
                tf.keras.layers.Conv1D(
                    filters=15, kernel_size=kernel_length, strides=5, padding='same', activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(
                    filters=15, kernel_size=kernel_length, strides=5, padding='same', activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(
                    filters=15, kernel_size=kernel_length, strides=5, padding='same', activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten()
            ]
        )
        self.emb_dims = num_filters[2]*2
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, self.emb_dims)),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):

        for n in range(self.sequence_length):
            x_input = x[:,n]
            if n==0:
                embeddings = self.cnn(x_input)
                embeddings = tf.expand_dims(embeddings,axis=1)
            else:
                cnn_out = self.cnn(x_input)
                embeddings = tf.concat((embeddings,tf.expand_dims(cnn_out,axis=1)),axis=1)
        out = self.rnn(embeddings)

        return out




class CRNN_2(tf.keras.Model):

    def __init__(self, kernel_length, num_filters, sequence_length, dropout):
        super(CRNN_2, self).__init__()
        self.sequence_length = sequence_length
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 1)),
                tf.keras.layers.Conv1D(
                    filters=num_filters[0], kernel_size=kernel_length, padding='same', activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool1D(pool_size=4),
                tf.keras.layers.Conv1D(
                    filters=num_filters[1], kernel_size=kernel_length, padding='same', activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool1D(pool_size=4),
                tf.keras.layers.Conv1D(
                    filters=num_filters[2], kernel_size=kernel_length, padding='same', activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool1D(pool_size=4),
                tf.keras.layers.Flatten()
            ]
        )
        self.emb_dims = num_filters[2]*2
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, self.emb_dims)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):

        for n in range(self.sequence_length):
            x_input = x[:,n]
            if n==0:
                embeddings = self.cnn(x_input)
                embeddings = tf.expand_dims(embeddings,axis=1)
            else:
                cnn_out = self.cnn(x_input)
                embeddings = tf.concat((embeddings,tf.expand_dims(cnn_out,axis=1)),axis=1)
        out = self.rnn(embeddings)

        return out



class CRNN_3(tf.keras.Model):

    def __init__(self, kernel_length, num_filters, sequence_length, dropout):
        super(CRNN_3, self).__init__()
        self.sequence_length = sequence_length
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 1)),
                tf.keras.layers.Conv1D(
                    filters=num_filters[0], kernel_size=kernel_length, padding='same', activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool1D(pool_size=4),
                tf.keras.layers.Conv1D(
                    filters=num_filters[1], kernel_size=kernel_length, padding='same', activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool1D(pool_size=4),
                tf.keras.layers.Conv1D(
                    filters=num_filters[2], kernel_size=kernel_length, padding='same', activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool1D(pool_size=4),
                tf.keras.layers.Flatten()
            ]
        )
        self.emb_dims = num_filters[2]*2
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(sequence_length, self.emb_dims)),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):

        for n in range(self.sequence_length):
            x_input = x[:,n]
            if n==0:
                embeddings = self.cnn(x_input)
                embeddings = tf.expand_dims(embeddings,axis=1)
            else:
                cnn_out = self.cnn(x_input)
                embeddings = tf.concat((embeddings,tf.expand_dims(cnn_out,axis=1)),axis=1)
        out = self.rnn(embeddings)

        return out


    
class CRNN_1_Timing(tf.keras.Model):

    def __init__(self, kernel_length, num_filters, sequence_length, dropout):
        super(CRNN_1_Timing, self).__init__()
        self.sequence_length = sequence_length
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 1)),
                tf.keras.layers.Conv1D(
                    filters=num_filters[0], kernel_size=kernel_length, padding='same', activation=None),
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool1D(pool_size=4),
                tf.keras.layers.Conv1D(
                    filters=num_filters[1], kernel_size=kernel_length, padding='same', activation=None),
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool1D(pool_size=4),
                tf.keras.layers.Conv1D(
                    filters=num_filters[2], kernel_size=kernel_length, padding='same', activation=None),
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool1D(pool_size=4),
                tf.keras.layers.Flatten()
            ]
        )
        self.emb_dims = num_filters[2]*2
        self.rnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.sequence_length, self.emb_dims)),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x):

        for n in range(self.sequence_length):
            x_input = x[:,n]
            if n==0:
                embeddings = self.cnn(x_input)
                embeddings = tf.expand_dims(embeddings,axis=1)
                embeddings_dummy = tf.zeros_like(embeddings)
            else:
                embeddings = tf.concat((embeddings,embeddings_dummy),axis=1)
        out = self.rnn(embeddings)

        return out

