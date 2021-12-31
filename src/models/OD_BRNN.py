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

    sequence_flat = np.zeros(sequence.size*factor)
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

    
num_crosstest = 10
factor_val = 0.15

n_epochs = 10000
patience_lr = 10
patience_early = 20

sequence_length = 20
factor_div = 5

num_thresholds_F1_score = 100.
min_sep = 3

lr = 1e-3
batch_size = 128

dropouts = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

for a in range(len(dropouts)):

    dropout = dropouts[a]

    Tensor_All = np.load('../../data/interim/Dataset_All.npy')
    Classes_All = np.load('../../data/interim/Classes_All.npy')
    
    for i in range(len(Classes_All)):
        if Classes_All[i]==1:
            Classes_All[i-1]==0.2
            Classes_All[i+1]==0.5
            Classes_All[i+2]==0.1

    fix_seeds(0)

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

    Tensor_All = (Tensor_All-np.min(Dataset_Train))/(np.max(Dataset_Train)-np.min(Dataset_Train)+1e-16)

    Tensor_All_Reduced = np.sum(Tensor_All, axis=1)
    Classes_All_Reduced = np.clip(np.sum(Classes_All, axis=1), 0, 1)

    skf = KFold(n_splits=num_crosstest)
    g = 0

    validation_accuracy = 0
    test_accuracy = 0

    for train_index, test_index in skf.split(Tensor_All_Reduced, Classes_All_Reduced):

        Tensor_Train_Val, Tensor_Test = Tensor_All[train_index], Tensor_All[test_index]
        Classes_Train_Val, Classes_Test = Classes_All[train_index], Classes_All[test_index]

        Tensor_Train = Tensor_Train_Val[:int(0.15*Tensor_Train_Val.shape[0])]
        Tensor_Val = Tensor_Train_Val[int(0.15*Tensor_Train_Val.shape[0]):]

        Classes_Train = Classes_Train_Val[:int(0.15*Tensor_Train_Val.shape[0])]
        Classes_Val = Classes_Train_Val[int(0.15*Tensor_Train_Val.shape[0]):]

        Dataset_Train = Dataset_Train.astype('float32')
        Dataset_Val = Dataset_Val.astype('float32')
        Dataset_Test = Dataset_Test.astype('float32')

        Classes_Train = Classes_Train.astype('float32')
        Classes_Val = Classes_Val.astype('float32')
        Classes_Test = Classes_Test.astype('float32')

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_early, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_lr)

        with tf.device(gpu_name):
            model = BRNN(sequence_length, dropout)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.MeanSquaredError(), metrics=['loss'])
            history = model.fit(Dataset_Train, Classes_Train, batch_size=batch_size, epochs=epochs, validation_data=(Dataset_Val, Classes_Val), callbacks=[early_stopping,lr_scheduler], shuffle=True)

        # Calculate validation threshold

        predictions = model.predict(Dataset_Val)
  
        Classes_Val[Classes_Val==0.1] = 0
        Classes_Val[Classes_Val==0.5] = 0
        Classes_Val[Classes_Val==0.2] = 0

        hop_size_ms = hop_size/22050

        #Prediction = flatten_sequence(sig(predictions), factor_div)
        Prediction = flatten_sequence(predictions, factor_div)
        Target = flatten_sequence(Classes_Val, factor_div)

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

        threshold = Threshold[f1_score.argmax()]
        validation_accuracy += np.max(f1_score)

        print('Test Accuracy: {:.4f}'.format(np.max(f1_score)))

        # Test

        predictions = model.predict(Dataset_Test)
  
        Classes_Test[Classes_Test==0.1] = 0
        Classes_Test[Classes_Test==0.5] = 0
        Classes_Test[Classes_Test==0.2] = 0

        hop_size_ms = hop_size/22050

        #Prediction = flatten_sequence(sig(predictions), factor_div)
        Prediction = flatten_sequence(predictions, factor_div)
        Target = flatten_sequence(Classes_Test, factor_div)

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

    print('')
    print('Dropout: ' + str(dropout))
    print('Mean Validation Accuracy: {:.4f}'.format(validation_accuracy/num_crosstest))
    print('Mean Test Accuracy: {:.4f}'.format(test_accuracy/num_crosstest))
    print('')