import os
import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
from itertools import combinations
import tensorflow_probability as tfp
from sklearn.model_selection import StratifiedShuffleSplit

from networks import *
from utils import *



'''os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.nice(0)
gpu_name = '/GPU:0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)'''

# Global parameters

percentage_train = 80

num_crossval = 5
num_iterations = 5

modes = ['E2E']

# Data parameters

frame_size = '1024'

# Training parameters

epochs = 10000
batch_size = 16

# Spectrogram normalisation values

norm_min_max = [[0.0, 3.7073483668036347],[-9.210340371976182, 9.999500033329732e-05]]

# Main loop

accuracies = np.zeros((28,num_iterations,num_crossval))

for m in range(len(modes)):

    mode = modes[m]

    if not os.path.isdir('models/' + mode):
        os.mkdir('models/' + mode)

    if not os.path.isdir('results/' + mode):
        os.mkdir('results/' + mode)

    for part in range(28):

        for it in range(num_iterations):

            Dataset_Train = np.load('data/interim/AVP_E2E/Dataset_Train_Aug_' + str(part).zfill(2)  + '.npy')
            Dataset_Test = np.load('data/interim/AVP_E2E/Dataset_Test_' + str(part).zfill(2)  + '.npy')

            # Spectrogram normalisation

            #Dataset_Train = (Dataset_Train-np.min(Dataset_Train))/(np.max(Dataset_Train)-np.min(Dataset_Train)+1e-16)
            Dataset_Train = np.log(Dataset_Train+1e-4)
            min_train = np.min(Dataset_Train)
            max_train = np.max(Dataset_Train)
            Dataset_Train = (Dataset_Train-min_train)/(max_train-min_train+1e-16)

            #Dataset_Test = (Dataset_Test-np.min(Dataset_Train))/(np.max(Dataset_Train)-np.min(Dataset_Train)+1e-16)
            Dataset_Test = np.log(Dataset_Test+1e-4)
            Dataset_Test = (Dataset_Test-min_train)/(max_train-min_train+1e-16)

            Dataset_Test = np.expand_dims(Dataset_Test,axis=-1).astype('float32')

            # Load and process classes
            
            Classes_Train_Str = np.load('data/interim/AVP_E2E/Classes_Train_Aug_' + str(part).zfill(2) + '.npy')
            Classes_Test_Str = np.load('data/interim/AVP_E2E/Classes_Test_' + str(part).zfill(2) + '.npy')

            Classes_Train = np.zeros(len(Classes_Train_Str))
            for n in range(len(Classes_Train_Str)):
                if Classes_Train_Str[n]=='kd':
                    Classes_Train[n] = 0
                elif Classes_Train_Str[n]=='sd':
                    Classes_Train[n] = 1
                elif Classes_Train_Str[n]=='hhc':
                    Classes_Train[n] = 2
                elif Classes_Train_Str[n]=='hho':
                    Classes_Train[n] = 3

            Classes_Test = np.zeros(len(Classes_Test_Str))
            for n in range(len(Classes_Test_Str)):
                if Classes_Test_Str[n]=='kd':
                    Classes_Test[n] = 0
                elif Classes_Test_Str[n]=='sd':
                    Classes_Test[n] = 1
                elif Classes_Test_Str[n]=='hhc':
                    Classes_Test[n] = 2
                elif Classes_Test_Str[n]=='hho':
                    Classes_Test[n] = 3

            num_classes = np.max(np.concatenate((Classes_Train,Classes_Test)))+1

            np.random.seed(0)
            np.random.shuffle(Dataset_Train)

            np.random.seed(0)
            np.random.shuffle(Dataset_Test)

            np.random.seed(0)
            np.random.shuffle(Classes_Train)

            np.random.seed(0)
            np.random.shuffle(Classes_Test)

            # Train models via 5-fold cross-validation (saving each model per fold)

            sss = StratifiedShuffleSplit(n_splits=num_crossval, test_size=0.2, random_state=0)
            Classes_Train_Split = Classes_Train.copy()

            cv = 0

            for train_index, test_index in sss.split(Dataset_Train, Classes_Train_Split):

                print('\n')
                print([part,it,cv])
                print('\n')

                Dataset_Train_Train, Dataset_Train_Val = Dataset_Train[train_index], Dataset_Train[test_index]

                Dataset_Train_Train = np.expand_dims(Dataset_Train_Train,axis=-1).astype('float32')
                Dataset_Train_Val = np.expand_dims(Dataset_Train_Val,axis=-1).astype('float32')

                Classes_Train_Train, Classes_Train_Val = Classes_Train[train_index], Classes_Train[test_index]

                Classes_Train_Train = Classes_Train_Train.astype('float32')
                Classes_Train_Val = Classes_Train_Val.astype('float32')

                patience_lr = 5
                patience_early = 10

                validation_accuracy = -1
                validation_loss = np.inf

                set_seeds(it)

                lr = 1e-3
                model = CNN_E2E_Big(num_classes)

                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience_early, restore_best_weights=False)
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=patience_lr)

                #with tf.device(gpu_name):
                with tf.device('cpu:0'):

                    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                    history = model.fit(Dataset_Train_Train, Classes_Train_Train, batch_size=batch_size, epochs=epochs, validation_data=(Dataset_Train_Val,Classes_Train_Val), callbacks=[early_stopping,lr_scheduler], shuffle=True, verbose=0)
                    test_loss, test_acc = model.evaluate(Dataset_Test, Classes_Test)

                model.save_weights('models/' + mode + '/pretrained_' + mode + '_' + str(part) + '_' + str(it) + '_' + str(cv) + '.h5')

                accuracies[part,it,cv] = test_acc

                tf.keras.backend.clear_session()

                cv += 1

    np.save('results/' + mode + '/accuracies_E2E', accuracies)

    # Calculate boxeme-wise weights

    num_test_boxemes = []
    for part in range(28):
        test_dataset = np.load('data/interim/AVP_E2E/Dataset_Test_' + str(part).zfill(2) + '.npy')
        num_test_boxemes.append(test_dataset.shape[0])
    boxeme_wise_weights = num_test_boxemes/np.sum(np.array(num_test_boxemes))

    # Results participant-wise

    print('\n')
    print('Participant-wise')
    print('\n')

    accuracies_raw = np.load('results/' + mode + '/accuracies.npy')
    accuracies_mean = np.mean(np.mean(accuracies_raw,axis=0))
    accuracies_std = np.std(np.mean(accuracies_raw,axis=0))
    print([mode])
    print([accuracies_mean,accuracies_std])

    # Results boxeme-wise

    print('\n')
    print('Boxeme-wise')
    print('\n')

    accuracies_raw = np.load('results/' + mode + '/accuracies.npy')
    for i in range(accuracies_raw.shape[1]):
        for j in range(accuracies_raw.shape[2]):
            accuracies_raw[:,i,j] *= boxeme_wise_weights*28
    accuracies_mean = np.mean(np.mean(accuracies_raw,axis=0))
    accuracies_std = np.std(np.mean(accuracies_raw,axis=0))
    print([mode])
    print([accuracies_mean,accuracies_std])
