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



'''os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.nice(0)
gpu_name = '/GPU:0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)'''

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



epochs = 15
patience_lr = 10
patience_early = 20

num_thresholds_F1_score = 100.

lr = 1e-4

mode = 'RNN_Stateful'

if not os.path.isdir('models/' + mode):
    os.mkdir('models/' + mode)
if not os.path.isdir('results/' + mode):
    os.mkdir('results/' + mode)

#networks = ['1','2','3']
networks = ['2']
sequence_lengths = [16]
eval_window_lengths = [0.0087,0.0145,0.0203,0.0261,0.0319]

dropout = 0

frame_dev_absmeans = np.zeros((len(sequence_lengths),len(networks),len(eval_window_lengths)))
frame_dev_absstds = np.zeros((len(sequence_lengths),len(networks),len(eval_window_lengths)))
frame_dev_means = np.zeros((len(sequence_lengths),len(networks),len(eval_window_lengths)))
frame_dev_stds = np.zeros((len(sequence_lengths),len(networks),len(eval_window_lengths)))

accuracies = np.zeros((len(sequence_lengths),len(networks),len(eval_window_lengths)))
precisions = np.zeros((len(sequence_lengths),len(networks),len(eval_window_lengths)))
recalls = np.zeros((len(sequence_lengths),len(networks),len(eval_window_lengths)))

all_thresholds_val = np.zeros((len(sequence_lengths),len(networks),len(eval_window_lengths)))

for a in range(len(sequence_lengths)):

    for b in range(len(networks)):

        sequence_length = sequence_lengths[a]
        network = networks[b]

        class ResetStatesCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super(ResetStatesCallback,self).__init__()
                self.counter = 0
            def on_batch_begin(self, batch, logs={}):
                if self.counter%sequence_length == 0:
                    self.model.reset_states()
                self.counter += 1

        set_seeds(0)

        Tensor_TrainVal = np.load('data/interim/Dataset_TrainVal.npy').T
        Classes_TrainVal = np.load('data/interim/Classes_TrainVal.npy')
        Tensor_Test = np.load('data/interim/Dataset_Test.npy').T
        Classes_Test = np.load('data/interim/Classes_Test.npy')
        
        for i in range(len(Classes_TrainVal)):
            if Classes_TrainVal[i]==1:
                Classes_TrainVal[i+1] = 0.999
                Classes_TrainVal[i+2] = 0.999

        set_seeds(0)

        #Tensor_TrainVal = np.lib.stride_tricks.sliding_window_view(Tensor_TrainVal,(sequence_length,Tensor_TrainVal.shape[1]))[:,0,:,:]
        #Tensor_Test = np.lib.stride_tricks.sliding_window_view(Tensor_Test,(sequence_length,Tensor_Test.shape[1]))[:,0,:,:]

        Tensor_TrainVal = Tensor_TrainVal[:-(Tensor_TrainVal.shape[0]%sequence_length)]
        Tensor_TrainVal = Tensor_TrainVal.reshape(Tensor_TrainVal.shape[0]//sequence_length,sequence_length,Tensor_TrainVal.shape[1])
        Tensor_Test = Tensor_Test[:-(Tensor_Test.shape[0]%sequence_length)]
        Tensor_Test = Tensor_Test.reshape(Tensor_Test.shape[0]//sequence_length,sequence_length,Tensor_Test.shape[1])

        Classes_TrainVal = Classes_TrainVal[:-(Classes_TrainVal.shape[0]%sequence_length)]
        Classes_TrainVal = Classes_TrainVal.reshape(Classes_TrainVal.shape[0]//sequence_length,sequence_length)
        Classes_Test = Classes_Test[:-(Classes_Test.shape[0]%sequence_length)]
        Classes_Test = Classes_Test.reshape(Classes_Test.shape[0]//sequence_length,sequence_length)

        Tensor_TrainVal = np.log(Tensor_TrainVal+1e-4)
        min_norm = np.min(Tensor_TrainVal)
        max_norm = np.max(Tensor_TrainVal)
        Tensor_TrainVal = (Tensor_TrainVal-min_norm)/(max_norm-min_norm+1e-16)
        Tensor_Test = np.log(Tensor_Test+1e-4)
        Tensor_Test = (Tensor_Test-min_norm)/(max_norm-min_norm+1e-16)

        Tensor_TrainVal_Reduced = np.sum(Tensor_TrainVal, axis=1)
        Classes_TrainVal_Reduced = np.clip(Classes_TrainVal, 0, 1)
        Tensor_Test_Reduced = np.sum(Tensor_Test, axis=1)
        Classes_Test_Reduced = np.clip(Classes_Test, 0, 1)

        Dataset_Test = Tensor_Test.copy()

        Dataset_Test = Dataset_Test.astype('float32')
        Classes_Test = Classes_Test.astype('float32')

        thresholds_val = np.zeros(len(eval_window_lengths))
        accuracies_val = np.zeros(len(eval_window_lengths))

        set_seeds(0)

        models = []

        Dataset_Train = Tensor_TrainVal[:-int(0.20*Tensor_TrainVal.shape[0])]
        Dataset_Val = Tensor_TrainVal[-int(0.20*Tensor_TrainVal.shape[0]):]
        Classes_Train = Classes_TrainVal[:-int(0.20*Tensor_TrainVal.shape[0])]
        Classes_Val = Classes_TrainVal[-int(0.20*Tensor_TrainVal.shape[0]):]

        Dataset_Train = Dataset_Train.astype('float32')
        Dataset_Val = Dataset_Val.astype('float32')

        Classes_Train = Classes_Train.astype('float32')
        Classes_Val = Classes_Val.astype('float32')

        #with tf.device(gpu_name):

        set_seeds(0)

        if network=='1':
            model = RNN_Stateful_1(sequence_length, dropout)
        elif network=='2':
            model = RNN_Stateful_2(sequence_length, dropout)
        elif network=='3':
            model = RNN_Stateful_3(sequence_length, dropout)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.MeanSquaredError()])

        Dataset_Train = np.expand_dims(Dataset_Train.reshape(Dataset_Train.shape[0]*Dataset_Train.shape[1],Dataset_Train.shape[2]),axis=-2)
        Classes_Train = Classes_Train.flatten()

        for epoch in range(epochs):
            history = model.fit(Dataset_Train, Classes_Train, epochs=1, callbacks=[ResetStatesCallback()], batch_size=1, shuffle=False)
            models.append(model)

        print('Processing validation data...')

        Classes_Val[Classes_Val!=1] = 0
        hop_size_ms = sequence_length/22050

        predictions = []
        for i in range(len(Dataset_Val)):
            if i%(len(Dataset_Val)//10)==0:
                print(str(i) + 'of' + str(len(Dataset_Val)))
            for j in range(sequence_length):
                predictions.append(model.predict_on_batch(np.expand_dims(np.expand_dims(Dataset_Val[i,j],axis=0),axis=0))[0][0])
            model.reset_states()
        
        Prediction = tf.math.sigmoid(predictions)
        Target = Classes_Val.reshape(Classes_Val.size)

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

        max_val_accuracies = np.zeros(len(eval_window_lengths))
        f1_score = np.zeros((len(eval_window_lengths),len(Threshold)))
        precision = np.zeros((len(eval_window_lengths),len(Threshold)))
        recall = np.zeros((len(eval_window_lengths),len(Threshold)))
        for n in range(len(eval_window_lengths)):
            print('Calculating threshold for evaluation window length = ' + str(eval_window_lengths[n]))
            for i in range(len(Threshold)):
                #Predicted = [1 if item>Threshold[i] else 0 for item in Prediction]
                #Predicted = np.array(Predicted)*factor
                #j = np.where(Predicted!=0)[0]
                #Pred = Prediction[j]
                Pred = np.argwhere(Prediction>=Threshold[i])[:,0]*hop_size_ms
                ind_delete = [i+1 for (x,y,i) in zip(Pred,Pred[1:],range(len(Pred))) if eval_window_lengths[n]/2>abs(x-y)]
                Pred = np.delete(Pred, ind_delete)
                for s in range(len(Pred)-1):
                    if Pred[s+1]<Pred[s]:
                        print('Ensuring Monotonic Predictions')
                        Pred[s+1] = Pred[s]
                f1_score[n,i], precision[n,i], recall[n,i] = f_measure(Target, Pred, window=eval_window_lengths[n])
            accuracies_val[n] = np.max(f1_score[n])
            thresholds_val[n] = Threshold[f1_score[n].argmax()]
            max_val_accuracies[n] = np.max(f1_score[n])

        print('Val Accuracies: ' + str(accuracies_val[n]))

        # Test

        print('Evaluating...')

        predictions = []
        for i in range(len(Dataset_Test)):
            if i%(len(Dataset_Test)//10)==0:
                print(str(i) + 'of' + str(len(Dataset_Test)))
            for j in range(sequence_length):
                predictions.append(model.predict_on_batch(np.expand_dims(np.expand_dims(Dataset_Test[i,j],axis=0),axis=0))[0][0])
            model.reset_states()

        Prediction = tf.math.sigmoid(predictions)
        Target = Classes_Test.reshape(Classes_Test.size)

        factor = np.arange(len(Target))*hop_size_ms
        Target = factor*Target

        j = np.where(Target!=0)
        Target = Target[j]
        
        Target = Target[:Target.argmax()]

        for s in range(len(Target)-1):
            if Target[s+1]<Target[s]:
                print('Ensuring Monotonic Target')
                Target[s+1] = Target[s]

        min_values = [[],[],[],[],[],[]]
        min_indices = [[],[],[],[],[],[]]
        for n in range(len(eval_window_lengths)):
            #Predicted = [1 if item>Threshold[i] else 0 for item in Prediction]
            #Predicted = np.array(Predicted)*factor
            #j = np.where(Predicted!=0)[0]
            #Pred = Prediction[j]
            Pred = np.argwhere(Prediction>=np.mean(thresholds_val[n]))[:,0]*hop_size_ms
            ind_delete = [i+1 for (x,y,i) in zip(Pred,Pred[1:],range(len(Pred))) if eval_window_lengths[n]/2>abs(x-y)]
            Pred = np.delete(Pred, ind_delete)
            for s in range(len(Pred)-1):
                if Pred[s+1]<Pred[s]:
                    print('Ensuring Monotonic Predictions')
                    Pred[s+1] = Pred[s]
            test_accuracy, test_precision, test_recall = f_measure(Target, Pred, window=eval_window_lengths[n])
            for k in range(len(Pred)):
                abs_diff = Target-Pred[k]
                diff = np.abs(abs_diff)
                if diff.argmin() not in min_indices[n]:
                    min_indices[n].append(diff.argmin())
                else:
                    continue
                min_value = abs_diff[diff.argmin()]
                if abs(min_value)<=eval_window_lengths[n]/2:
                    min_values[n].append(min_value)

            min_values[n] = np.array(min_values[n])

            frame_dev_absmeans[a,b,n] = np.mean(np.abs(min_values[n]))
            frame_dev_absstds[a,b,n] = np.std(np.abs(min_values[n]))
            frame_dev_means[a,b,n] = np.mean(min_values[n])
            frame_dev_stds[a,b,n] = np.std(min_values[n])
            
            accuracies[a,b,n] = test_accuracy
            precisions[a,b,n] = test_precision
            recalls[a,b,n] = test_recall
            all_thresholds_val[a,b,n] = thresholds_val[n]

            print('')

            print('Hop Size: ' + str(sequence_length))
            
            print('Mean Absolute Onset Deviation: ' + str(frame_dev_absmeans[a,b,n]))
            print('STD Absolute Onset Deviation: ' + str(frame_dev_absstds[a,b,n]))
            print('Mean Deviation: ' + str(frame_dev_means[a,b,n]))
            print('STD Deviation: ' + str(frame_dev_stds[a,b,n]))
            
            print('Accuracy: ' + str(accuracies[a,b,n]))
            print('Precision: ' + str(precisions[a,b,n]))
            print('Recall: ' + str(recalls[a,b,n]))
            print('Cross-Validated Thresholds: ' + str(all_thresholds_val[a,b,n]))

            print('')

            np.save('results/' + mode + '/frame_dev_absstds', frame_dev_absstds)
            np.save('results/' + mode + '/frame_dev_absmeans', frame_dev_absmeans)
            np.save('results/' + mode + '/frame_dev_means', frame_dev_means)
            np.save('results/' + mode + '/frame_dev_stds', frame_dev_stds)

            np.save('results/' + mode + '/accuracies', accuracies)
            np.save('results/' + mode + '/precisions', precisions)
            np.save('results/' + mode + '/recalls', recalls)
            np.save('results/' + mode + '/thresholds', all_thresholds_val)

            #if accuracies[a,n]==np.max(accuracies[:,n]):
                #model.save_weights('models/' + mode + '/model_window_' + str(eval_window_lengths[n]) + '.h5')
