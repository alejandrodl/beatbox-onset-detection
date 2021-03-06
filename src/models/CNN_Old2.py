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



os.environ["CUDA_VISIBLE_DEVICES"]="0"
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



mode = 'CNN'
    
num_crosstest = 8
factor_val = 0.15

epochs = 10000
patience_lr = 10
patience_early = 20

sequence_length = 16
factor_div = 4

num_thresholds_F1_score = 100.
min_sep = 3

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

mean_accuracies = np.zeros(len(dropouts))
std_accuracies = np.zeros(len(dropouts))
mean_precisions = np.zeros(len(dropouts))
std_precisions = np.zeros(len(dropouts))
mean_recalls = np.zeros(len(dropouts))
std_recalls = np.zeros(len(dropouts))

for a in range(len(dropouts)):

    dropout = dropouts[a]

    set_seeds(0)

    Tensor_All = np.load('../../data/interim/Dataset_All.npy').T
    Classes_All = np.load('../../data/interim/Classes_All.npy')

    for i in range(len(Classes_All)):
        if Classes_All[i]==1:
            Classes_All[i-1]==0.2
            Classes_All[i+1]==0.5
            Classes_All[i+2]==0.1

    cut_length = int(Tensor_All.shape[0]/sequence_length)*sequence_length

    Classes_All_0 = Classes_All[:cut_length]
    Tensor_All_0 = Tensor_All[:cut_length]

    div_sequence_length = sequence_length//factor_div

    Tensor_All = np.zeros((int(Tensor_All_0.shape[0]/div_sequence_length)-(factor_div-1), sequence_length, Tensor_All_0.shape[1]))
    Classes_All = np.zeros((int(len(Classes_All_0)/div_sequence_length)-(factor_div-1), sequence_length))

    for n in range(int(Tensor_All_0.shape[0]/div_sequence_length)-(factor_div-1)):
        point = n*div_sequence_length
        Tensor_All[n] = Tensor_All_0[point:point+sequence_length]
        Classes_All[n] = Classes_All_0[point:point+sequence_length]

    Tensor_All = np.log(Tensor_All+1e-4)
    Tensor_All = (Tensor_All-np.min(Tensor_All))/(np.max(Tensor_All)-np.min(Tensor_All)+1e-16)

    Tensor_All_Reduced = np.sum(Tensor_All, axis=1)
    Classes_All_Reduced = np.clip(np.sum(Classes_All, axis=1), 0, 1)

    skf = KFold(n_splits=num_crosstest)

    validation_accuracy = 0
    test_accuracy = 0

    validation_accuracies = np.zeros(num_crosstest)
    validation_precisions = np.zeros(num_crosstest)
    validation_recalls = np.zeros(num_crosstest)

    test_accuracies = np.zeros(num_crosstest)
    test_precisions = np.zeros(num_crosstest)
    test_recalls = np.zeros(num_crosstest)

    set_seeds(0)

    models = []
    g = 0

    for train_index, test_index in skf.split(Tensor_All_Reduced, Classes_All_Reduced):

        Dataset_Train_Val, Dataset_Test = Tensor_All[train_index], Tensor_All[test_index]
        Classes_Train_Val, Classes_Test = Classes_All[train_index], Classes_All[test_index]

        Dataset_Train = Dataset_Train_Val[:-int(0.15*Dataset_Train_Val.shape[0])]
        Dataset_Val = Dataset_Train_Val[-int(0.15*Dataset_Train_Val.shape[0]):]

        Classes_Train = Classes_Train_Val[:-int(0.15*len(Classes_Train_Val))]
        Classes_Val = Classes_Train_Val[-int(0.15*len(Classes_Train_Val)):]

        Dataset_Train = np.expand_dims(Dataset_Train,axis=-1).astype('float32')
        Dataset_Val = np.expand_dims(Dataset_Val,axis=-1).astype('float32')
        Dataset_Test = np.expand_dims(Dataset_Test,axis=-1).astype('float32')

        Classes_Train = Classes_Train.astype('float32')
        Classes_Val = Classes_Val.astype('float32')
        Classes_Test = Classes_Test.astype('float32')

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=patience_early, verbose=2, mode='auto', baseline=None, restore_best_weights=False)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_lr, min_delta=1e-4, verbose=2)

        with tf.device(gpu_name):
            set_seeds(0)
            model = CNN_T(sequence_length, dropout)
            set_seeds(0)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)) # , metrics=['accuracy']
            set_seeds(0)
            history = model.fit(Dataset_Train, Classes_Train, batch_size=batch_size, epochs=epochs, validation_data=(Dataset_Val, Classes_Val), callbacks=[early_stopping,lr_scheduler], shuffle=True)

        models.append(model)

        # Calculate threshold parameter with train-validation set

        print('Loading train-val data...')

        predictions = model.predict(np.expand_dims(Dataset_Train_Val,axis=-1).astype('float32'))
  
        Classes_Train_Val[Classes_Train_Val==0.1] = 0
        Classes_Train_Val[Classes_Train_Val==0.5] = 0
        Classes_Train_Val[Classes_Train_Val==0.2] = 0

        print('Preparing train-val data...')

        hop_size_ms = hop_size/22050

        Prediction = flatten_sequence(tf.math.sigmoid(predictions), factor_div)
        Target = flatten_sequence(Classes_Train_Val, factor_div)

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

        threshold = Threshold[f1_score.argmax()]

        # Test

        print('Evaluating...')

        predictions = model.predict(Dataset_Test.astype('float32'))
  
        Classes_Test[Classes_Test==0.1] = 0
        Classes_Test[Classes_Test==0.5] = 0
        Classes_Test[Classes_Test==0.2] = 0

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

        Predicted = [1 if item>threshold else 0 for item in Prediction]
        Predicted = np.array(Predicted)*factor
        j = np.where(Predicted!=0)
        Pred = Predicted[j]
        ind_delete = [i+1 for (x,y,i) in zip(Pred,Pred[1:],range(len(Pred))) if 0.015>abs(x-y)]
        Pred = np.delete(Pred, ind_delete)
        test_accuracy, test_precision, test_recall = f_measure(Target, Pred, window=0.03)

        print('Test Accuracy: {:.4f}'.format(test_accuracy))

        if g==0:
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

        test_accuracies[g] = test_accuracy
        test_precisions[g] = test_precision
        test_recalls[g] = test_recall
        g += 1
        
    min_value = np.array(min_value)
    
    frame_dev_absmeans[a] = np.mean(np.abs(min_values))
    frame_dev_absstds[a] = np.mean(np.abs(min_values))
    frame_dev_means[a] = np.mean(min_values)
    frame_dev_stds[a] = np.std(min_values)
    
    mean_accuracies[a] = np.mean(test_accuracies)
    std_accuracies[a] = np.std(test_accuracies)
    mean_precisions[a] = np.mean(test_precisions)
    std_precisions[a] = np.std(test_precisions)
    mean_recalls[a] = np.mean(test_recalls)
    std_recalls[a] = np.std(test_recalls)

    print('')

    print('Dropout: ' + str(dropout))
    
    print('Mean Absolute Onset Deviation All: ' + str(frame_dev_absmeans[a]))
    print('STD Absolute Onset Deviation All: ' + str(frame_dev_absstds[a]))
    print('Mean Deviation All: ' + str(frame_dev_means[a]))
    print('STD Deviation All: ' + str(frame_dev_stds[a]))
    
    print('Mean Accuracy: ' + str(mean_accuracies[a]))
    print('STD Accuracy: ' + str(std_accuracies[a]))
    print('Mean Precision: ' + str(mean_precisions[a]))
    print('STD Precision: ' + str(std_precisions[a]))
    print('Mean Recall: ' + str(mean_recalls[a]))
    print('STD Recall: ' + str(std_recalls[a]))

    print('')

    np.save('../../results/' + mode + '/frame_dev_absstds', frame_dev_absstds)
    np.save('../../results/' + mode + '/frame_dev_absmeans', frame_dev_absmeans)
    np.save('../../results/' + mode + '/frame_dev_means', frame_dev_means)
    np.save('../../results/' + mode + '/frame_dev_stds', frame_dev_stds)

    np.save('../../results/' + mode + '/mean_accuracies', mean_accuracies)
    np.save('../../results/' + mode + '/std_accuracies', std_accuracies)
    np.save('../../results/' + mode + '/mean_precisions', mean_precisions)
    np.save('../../results/' + mode + '/std_precisions', std_precisions)
    np.save('../../results/' + mode + '/mean_recalls', mean_recalls)
    np.save('../../results/' + mode + '/std_recalls', std_recalls)

    if mean_accuracies[a]==np.max(mean_accuracies):
        for cv in range(num_crosstest):
            models[cv].save_weights('../../models/' + mode + '/model_dropout_' + str(dropout) + '_crossval' + str(cv) + '.h5')