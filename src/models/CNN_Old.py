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



os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.nice(0)
gpu_name = '/GPU:0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def Make_KFolds_CNN(dataset, classes, k):

    c = 0
    indices = []
    for n in range(len(classes)):
        if np.sum(classes[n])==0:
            if indices and n-1 not in indices[c-1]:
                indices.append([indices[c-1][1],n])
                c += 1
            elif not indices:
                indices.append([0,n])
                c += 1

    if indices[0]==[0,0]:
        indices = indices[1:]
    if indices[-1][1]<len(classes):
        indices.append([indices[-1][1],len(classes)])

    length_folds = len(classes)//k

    classes_folds = []
    dataset_folds = []

    n_start = 0
    start_point = 0
    ending_point = indices[-1][1]

    for i in range(k):

        classes_fold = []
        dataset_fold = []

        for n in range(n_start,len(indices)):
            difference = indices[n][1] - indices[n][0]
            classes_fold.append(classes[indices[n][0]:indices[n][1]])
            dataset_fold.append(dataset[indices[n][0]:indices[n][1]])
            start_point += difference
            if indices[n][1]>=(i+1)*length_folds:
                n_start = n+1
                break

        classes_folds.append(np.concatenate(classes_fold))
        dataset_folds.append(np.concatenate(dataset_fold))
        
    return dataset_folds, classes_folds


    
num_crosstest = 10
factor_val = 0.15

n_epochs = 10000
patience_lr = 10
patience_early = 20

time_length = 16
num_thresholds_F1_score = 100.
min_sep = 3

lr = 1e-3
batch_size = 128

dropouts = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

for a in range(len(dropouts)):

    dropout = dropouts[a]

    Tensor_All = np.load('../../data/interim/Dataset_All.npy').T
    Classes_All = np.load('../../data/interim/Classes_All.npy')
    
    for i in range(len(Classes_All)):
        if Classes_All[i]==1:
            Classes_All[i-1]==0.2
            Classes_All[i+1]==0.5
            Classes_All[i+2]==0.1

    set_seeds(0)

    zp = np.zeros((time_length//2, Tensor_All.shape[-1]))
    Tensor_All_0 = np.concatenate((zp, Tensor_All, zp))

    Tensor_All = np.zeros((Tensor_All_0.shape[0]-time_length, time_length, Tensor_All_0.shape[1]))
    for n in range(Tensor_All.shape[0]):
        Tensor_All[n] = Tensor_All_0[n:n+time_length]

    Tensor_All = (Tensor_All-np.min(Tensor_All))/(np.max(Tensor_All)-np.min(Tensor_All)+1e-16)

    Tensor_All, Classes_All = Make_KFolds_CNN(Tensor_All, Classes_All, num_crosstest)

    Tensor_All_Reduced = np.sum(Tensor_All, axis=1)
    Classes_All_Reduced = np.clip(np.sum(Classes_All, axis=1), 0, 1)

    skf = KFold(n_splits=num_crosstest)

    validation_accuracy = 0
    test_accuracy = 0

    for g in range(num_crossval):
        
        indices_train = np.delete(np.arange(num_crossval),g)
        
        for n in range(1,len(indices_train)):
            if n==1:
                Tensor_Train_Val = np.vstack((Tensor_All[indices_train[0]],Tensor_All[indices_train[1]]))
                Labels_Train_Val = np.concatenate((Labels_All[indices_train[0]],Labels_All[indices_train[1]])) 
            else:
                Tensor_Train_Val = np.vstack((Tensor_Train_Val, Tensor_All[indices_train[n]]))
                Labels_Train_Val = np.concatenate((Labels_Train_Val, Labels_All[indices_train[n]]))
                
        Tensor_Test = Tensor_All[g]
        Labels_Test = Labels_All[g]

        cutoff_index = -int(0.15*Tensor_Train_Val.shape[0])

        Tensor_Train = Tensor_Train_Val[:cutoff_index]
        Tensor_Val = Tensor_Train_Val[cutoff_index:]

        Labels_Train = Labels_Train_Val[:cutoff_index]
        Labels_Val = Labels_Train_Val[cutoff_index:]

        Dataset_Train = np.expand_dims(Dataset_Train,axis=-1).astype('float32')
        Dataset_Val = np.expand_dims(Dataset_Val,axis=-1).astype('float32')
        Dataset_Test = np.expand_dims(Dataset_Test,axis=-1).astype('float32')

        Classes_Train = Classes_Train.astype('float32')
        Classes_Val = Classes_Val.astype('float32')
        Classes_Test = Classes_Test.astype('float32')

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_early, verbose=2, mode='auto', baseline=None, restore_best_weights=False)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_lr, verbose=2)

        with tf.device(gpu_name):
            model = CNN(time_length, dropout)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)) # , metrics=['accuracy']
            history = model.fit(Dataset_Train, Classes_Train, batch_size=batch_size, epochs=epochs, validation_data=(Dataset_Val, Classes_Val), callbacks=[early_stopping,lr_scheduler], shuffle=True)

        # Calculate threshold parameter with train-validation set

        print('Loading train-val data...')

        predictions = model.predict(Dataset_Train_Val)
  
        Classes_Train_Val[Classes_Train_Val==0.1] = 0
        Classes_Train_Val[Classes_Train_Val==0.5] = 0
        Classes_Train_Val[Classes_Train_Val==0.2] = 0

        print('Preparing train-val data...')

        hop_size_ms = hop_size/22050

        factor = (np.arange(len(classes.flatten()))+1)*hop_size_ms
        Target = classes.flatten()*factor
        Prediction = tf.math.sigmoid(predictions).flatten()

        factor = np.arange(len(Target))*hop_size_ms
        Target = factor*Target

        j = np.where(Target!=0)
        Target = Target[j]
        
        Target = Target[:Target.argmax()]

        num_thresholds = num_thresholds_F1_score
        Threshold = np.arange(int(num_thresholds+2))/(num_thresholds+2)
        Threshold = Threshold[1:-1]

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
            print('Calculating threshold: ' + str(i+1) + '/' + str(len(Threshold)))

        threshold = Threshold[f1_score.argmax()]
        validation_accuracy += np.max(f1_score)

        print('Train-Validation Accuracy: {:.4f}'.format(np.max(f1_score)))

        # Test

        predictions = model.predict(Dataset_Test)
  
        Classes_Test[Classes_Test==0.1] = 0
        Classes_Test[Classes_Test==0.5] = 0
        Classes_Test[Classes_Test==0.2] = 0

        hop_size_ms = hop_size/22050

        factor = (np.arange(len(classes.flatten()))+1)*hop_size_ms
        Target = classes.flatten()*factor
        Prediction = tf.math.sigmoid(predictions).flatten()

        factor = np.arange(len(Target))*hop_size_ms
        Target = factor*Target

        j = np.where(Target!=0)
        Target = Target[j]
        
        Target = Target[:Target.argmax()]

        Predicted = [1 if item>threshold else 0 for item in Prediction]
        Predicted = np.array(Predicted)*factor
        j = np.where(Predicted!=0)
        Pred = Predicted[j]
        ind_delete = [i+1 for (x,y,i) in zip(Pred,Pred[1:],range(len(Pred))) if 0.015>abs(x-y)]
        Pred = np.delete(Pred, ind_delete)
        f1_score, precision, recall = f_measure(Target, Pred, window=0.03)

        test_accuracy += f1_score

        print('Test Accuracy: {:.4f}'.format(f1_score))

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

        test_accuracy /= float(count_batch_test)
        test_precision /= float(count_batch_test)
        test_recall /= float(count_batch_test)
        print('Test Accuracy: {:.4f}'.format(test_accuracy))

        test_accuracies[g] = test_accuracy
        test_precisions[g] = test_precision
        test_recalls[g] = test_recall
        g += 1
        
    min_values = np.array(min_values)
    
    frame_dev_median = np.median(min_values)
    frame_dev_mean = np.mean(min_values)
    frame_dev_std = np.std(min_values)
    
    mean_accuracy = np.mean(test_accuracies)
    mean_precision = np.mean(test_precisions)
    mean_recall = np.mean(test_recalls)

    print('')

    print('Dropout: ' + str(dropout))
    
    print('Median Deviation All: ' + str(frame_dev_median))
    print('Mean Deviation All: ' + str(frame_dev_mean))
    print('STD Deviation All: ' + str(frame_dev_std))
    
    print('Mean Accuracy: ' + str(mean_accuracy))
    print('Mean Precision: ' + str(mean_precision))
    print('Mean Recall: ' + str(mean_recall))

    print('')