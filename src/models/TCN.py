import os
import pdb
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics import f1_score
from mir_eval.onset import f_measure

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from networks import *
from utils import set_seeds

from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential



os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.nice(0)
gpu_name = '/GPU:0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def flatten_sequence(sequence, factor):
    
    seq_length = sequence.shape[-1]
    length = seq_length//factor
    seq_length_diff = seq_length - length

    sequence_flat = np.zeros(tf.size(Dataset_Test).numpy()*factor)
    for n in range(len(sequence)):
        point = n*length
        if n==0:
            sequence_flat[:seq_length] = sequence[n]
        else:
            sequence_flat[point:point+seq_length_diff] = sequence_flat[point:point+seq_length_diff] + sequence[n][:-length]
            sequence_flat[point+seq_length_diff:point+seq_length_diff+length] = sequence[n][-length:]

    sequence_flat = sequence_flat[:point+seq_length]
    
    for n in range(factor-1):
        point = n*length
        sequence_flat[point:point+length] = sequence_flat[point:point+length]/(n+1)
        if n==0:
            sequence_flat[-point-length:] = sequence_flat[-point-length:]/(n+1)
        else:
            sequence_flat[-point-length:-point] = sequence_flat[-point-length:-point]/(n+1)
        
    sequence_flat[(factor-1)*length:-(factor-1)*length] = sequence_flat[(factor-1)*length:-(factor-1)*length]/factor
    
    return sequence_flat

    

mode = 'BRNN_3'

num_crossval = 7

epochs = 10000
patience_lr = 10
patience_early = 20

sequence_length = 1024
factor_div = 4

num_thresholds_F1_score = 100.

lr = 1e-3
batch_size = 1024
hop_size = 128

dropouts = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
#dropouts = [0.3]

if not os.path.isdir('../../models/' + mode):
    os.mkdir('../../models/' + mode)

if not os.path.isdir('../../results/' + mode):
    os.mkdir('../../results/' + mode)

frame_dev_absmeans = np.zeros(len(dropouts))
frame_dev_absstds = np.zeros(len(dropouts))
frame_dev_means = np.zeros(len(dropouts))
frame_dev_stds = np.zeros(len(dropouts))

accuracies = np.zeros(len(dropouts))
precisions = np.zeros(len(dropouts))
recalls = np.zeros(len(dropouts))

all_thresholds_crossval = np.zeros((len(dropouts),num_crossval))

for a in range(len(dropouts)):

    dropout = dropouts[a]

    set_seeds(0)

    Tensor_TrainVal_Raw = np.load('../../data/interim/Dataset_TrainVal.npy').T
    Classes_TrainVal_Raw = np.load('../../data/interim/Classes_TrainVal.npy')
    Tensor_Test_Raw = np.load('../../data/interim/Dataset_Test.npy').T
    Classes_Test_Raw = np.load('../../data/interim/Classes_Test.npy')
    
    for i in range(len(Classes_TrainVal)):
        if Classes_TrainVal[i]==1:
            Classes_TrainVal[i-1] = 0.2
            Classes_TrainVal[i+1] = 0.5
            Classes_TrainVal[i+2] = 0.1

    set_seeds(0)

    #Tensor_TrainVal = np.lib.stride_tricks.sliding_window_view(Tensor_TrainVal,(sequence_length,Tensor_TrainVal.shape[1]))[:,0,:,:]
    #Tensor_Test = np.lib.stride_tricks.sliding_window_view(Tensor_Test,(sequence_length,Tensor_Test.shape[1]))[:,0,:,:]

    length = Tensor_TrainVal_Raw.shape[0]-sequence_length+1
    Tensor_TrainVal = np.zeros(shape=(length,sequence_length,Tensor_TrainVal_Raw.shape[1]))
    Classes_TrainVal = np.zeros(shape=(length,sequence_length))
    for n in range(sequence_length):
        Tensor_TrainVal[:,n] = Tensor_TrainVal_Raw[n:length+n]
        Classes_TrainVal[:,n] = Classes_TrainVal_Raw[n:length+n]
    Tensor_TrainVal = Tensor_TrainVal[::factor_div]
    Classes_TrainVal = Classes_TrainVal[::factor_div]

    length = Tensor_Test_Raw.shape[0]-sequence_length+1
    Tensor_Test = np.zeros(shape=(length,sequence_length,Tensor_Test_Raw.shape[1]))
    Classes_Test = np.zeros(shape=(length,sequence_length))
    for n in range(sequence_length):
        Tensor_Test[:,n] = Tensor_Test_Raw[n:length+n]
        Classes_Test[:,n] = Classes_Test_Raw[n:length+n]
    Tensor_Test = Tensor_Test[::factor_div]
    Classes_Test = Classes_Test[::factor_div]

    Tensor_TrainVal = np.log(Tensor_TrainVal+1e-4)
    min_norm = np.min(Tensor_TrainVal)
    max_norm = np.max(Tensor_TrainVal)
    Tensor_TrainVal = (Tensor_TrainVal-min_norm)/(max_norm-min_norm+1e-16)
    Tensor_Test = np.log(Tensor_Test+1e-4)
    Tensor_Test = (Tensor_Test-min_norm)/(max_norm-min_norm+1e-16)

    Tensor_TrainVal_Reduced = np.sum(Tensor_TrainVal, axis=1)
    Classes_TrainVal_Reduced = np.clip(np.sum(Classes_TrainVal, axis=1), 0, 1)
    Tensor_Test_Reduced = np.sum(Tensor_Test, axis=1)
    Classes_Test_Reduced = np.clip(np.sum(Classes_Test, axis=1), 0, 1)

    Dataset_Test = Tensor_Test.copy()

    Dataset_Test = Dataset_Test.astype('float32')
    Classes_Test = Classes_Test.astype('float32')

    thresholds_crossval = np.zeros(num_crossval)





# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.
batch_size = None
time_steps = 16
input_dim = 1

def get_x_y(size=1000):
    import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, time_steps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0  # we introduce the target in the first timestep of the sequence.
    y_train[pos_indices, 0] = 1.0  # the task is to see if the TCN can go back in time to find it.
    return x_train, y_train


tcn_layer = TCN(input_shape=(time_steps, input_dim))
# The receptive field tells you how far the model can see in terms of timesteps.
print('Receptive field size =', tcn_layer.receptive_field)

m = Sequential([
    tcn_layer,
    Dense(1)
])

m.compile(optimizer='adam', loss='mse')

tcn_full_summary(m, expand_residual_blocks=False)

x, y = get_x_y()
m.fit(x, y, epochs=10, validation_split=0.2)




























        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_early, verbose=2, mode='auto', baseline=None, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_lr, verbose=2)

        with tf.device(gpu_name):
            set_seeds(0)
            model = BRNN_3(sequence_length, dropout)
            set_seeds(0)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)) # , metrics=['accuracy']
            set_seeds(0)
            history = model.fit(Dataset_Train, Classes_Train, batch_size=batch_size, epochs=epochs, validation_data=(Dataset_Val, Classes_Val), callbacks=[early_stopping,lr_scheduler], shuffle=True)

        if min(history.history['val_loss'])<min_val_loss:
            idx_best_model = g
            min_val_loss = min(history.history['val_loss'])

        print('Val Loss for fold ' + str(g+1) + ' of ' + str(num_crossval) + ': ' + str(min(history.history['val_loss'])))

        models.append(model)

        # Calculate threshold parameter with validation set

        print('Loading validation data...')

        predictions = model.predict(Dataset_Val.astype('float32'))
  
        Classes_Val[Classes_Val!=1] = 0

        print('Preparing validation data...')

        hop_size_ms = hop_size/22050

        Prediction = flatten_sequence(tf.math.sigmoid(predictions), factor_div)
        Target = flatten_sequence(Classes_Val, factor_div)

        factor = np.arange(len(Target))*hop_size_ms
        Target = factor*Target

        j = np.where(Target!=0)
        Target = Target[j]
        
        Target = Target[:Target.argmax()]

        for s in range(len(Target)-1):
            if Target[s+1]<Target[s]:
                print('Ensuring Monotonic Target')
                Target[s+1] = Target[s]

        num_thresholds = num_thresholds_F1_score
        Threshold = np.arange(int(num_thresholds+2))/(num_thresholds+2)
        Threshold = Threshold[1:-1]

        print('Calculating threshold...')

        f1_score = np.zeros(len(Threshold))
        precision = np.zeros(len(Threshold))
        recall = np.zeros(len(Threshold))
        for i in range(len(Threshold)):
            Predicted = [1 if item>Threshold[i] else 0 for item in Prediction]
            Predicted = np.array(Predicted)*factor
            j = np.where(Predicted!=0)
            Pred = Predicted[j]
            ind_delete = [i+1 for (x,y,i) in zip(Pred,Pred[1:],range(len(Pred))) if 0.015>abs(x-y)]
            Pred = np.delete(Pred, ind_delete)
            f1_score[i], precision[i], recall[i] = f_measure(Target, Pred, window=0.03)

        print('Val Accuracy for fold ' + str(g+1) + ' of ' + str(num_crossval) + ': ' + str(np.max(f1_score)))
        thresholds_crossval[g] = Threshold[f1_score.argmax()]
        g += 1

    # Test

    print('Evaluating...')

    model = models[idx_best_model]

    predictions = model.predict(Dataset_Test.astype('float32'))

    hop_size_ms = hop_size/22050

    Prediction = flatten_sequence(tf.math.sigmoid(predictions), factor_div)
    Target = flatten_sequence(Classes_Test, factor_div)

    factor = np.arange(len(Target))*hop_size_ms
    Target = factor*Target

    j = np.where(Target!=0)
    Target = Target[j]
    
    Target = Target[:Target.argmax()]

    for s in range(len(Target)-1):
        if Target[s+1]<Target[s]:
            print('Ensuring Monotonic Target')
            Target[s+1] = Target[s]

    threshold = np.mean(thresholds_crossval)

    Predicted = [1 if item>threshold else 0 for item in Prediction]
    Predicted = np.array(Predicted)*factor
    j = np.where(Predicted!=0)
    Pred = Predicted[j]
    ind_delete = [i+1 for (x,y,i) in zip(Pred,Pred[1:],range(len(Pred))) if 0.015>abs(x-y)]
    Pred = np.delete(Pred, ind_delete)
    test_accuracy, test_precision, test_recall = f_measure(Target, Pred, window=0.03)

    print('Test Accuracy: {:.4f}'.format(test_accuracy))

    min_values = []
    min_indices = []
    for k in range(len(Pred)):
        abs_diff = Target-Pred[k]
        diff = np.abs(abs_diff)
        if diff.argmin() not in min_indices:
            min_indices.append(diff.argmin())
        else:
            continue
        min_value = abs_diff[diff.argmin()]
        if abs(min_value)<=0.015:
            min_values.append(min_value)
        
    min_values = np.array(min_values)
    
    frame_dev_absmeans[a] = np.mean(np.abs(min_values))
    frame_dev_absstds[a] = np.std(np.abs(min_values))
    frame_dev_means[a] = np.mean(min_values)
    frame_dev_stds[a] = np.std(min_values)
    
    accuracies[a] = test_accuracy
    precisions[a] = test_precision
    recalls[a] = test_recall
    all_thresholds_crossval[a] = thresholds_crossval

    print('')

    print('Dropout: ' + str(dropout))
    
    print('Mean Absolute Onset Deviation: ' + str(frame_dev_absmeans[a]))
    print('STD Absolute Onset Deviation: ' + str(frame_dev_absstds[a]))
    print('Mean Deviation: ' + str(frame_dev_means[a]))
    print('STD Deviation: ' + str(frame_dev_stds[a]))
    
    print('Accuracy: ' + str(accuracies[a]))
    print('Precision: ' + str(precisions[a]))
    print('Recall: ' + str(recalls[a]))
    print('Cross-Validated Thresholds: ' + str(all_thresholds_crossval[a]))

    print('')

    np.save('../../results/' + mode + '/frame_dev_absstds', frame_dev_absstds)
    np.save('../../results/' + mode + '/frame_dev_absmeans', frame_dev_absmeans)
    np.save('../../results/' + mode + '/frame_dev_means', frame_dev_means)
    np.save('../../results/' + mode + '/frame_dev_stds', frame_dev_stds)

    np.save('../../results/' + mode + '/accuracies', accuracies)
    np.save('../../results/' + mode + '/precisions', precisions)
    np.save('../../results/' + mode + '/recalls', recalls)
    np.save('../../results/' + mode + '/thresholds', all_thresholds_crossval)

    if accuracies[a]==np.max(accuracies):
        for g in range(num_crossval):
            models[idx_best_model].save_weights('../../models/' + mode + '/model_dropout_' + str(dropout) + '_crossval_' + str(idx_best_model) + '.h5')