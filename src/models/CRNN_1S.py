import os
import pdb
import random
import madmom
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

    

mode = 'CRNN_1S'

num_crossval = 7

epochs = 10000
patience_lr = 10
patience_early = 20

sequence_length_1 = 8
sequence_length_2 = 12
hop = 3

pre_avg = np.array([0])*0.01
post_avg = np.array([0])*0.01
pre_max = np.array([0,1,2,4,7])*0.01
post_max = np.array([0,1,2,4,7])*0.01

#pre_avg = np.array([0])
#post_avg = np.array([0])
#pre_max = np.array([0])
#post_max = np.array([0])

num_thresholds_F1_score = 100.

lr = 1e-3
batch_size = 1024
hop_size = 128

#dropouts = [0,0.05,0.1,0.15,0.2,0.25,0.3]
dropouts = [0.1]
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

    Tensor_TrainVal_Raw = (np.load('../../data/interim/Dataset_TrainVal.npy').T).astype(np.float32)
    Classes_TrainVal_Raw = np.load('../../data/interim/Classes_TrainVal.npy').astype(np.float32)
    Tensor_Test_Raw = (np.load('../../data/interim/Dataset_Test.npy').T).astype(np.float32)
    Classes_Test_Raw = np.load('../../data/interim/Classes_Test.npy').astype(np.float32)
    
    for i in range(len(Classes_TrainVal_Raw)):
        if Classes_TrainVal_Raw[i]==1:
            Classes_TrainVal_Raw[i-1] = 0.2
            Classes_TrainVal_Raw[i+1] = 0.5
            Classes_TrainVal_Raw[i+2] = 0.1

    set_seeds(0)

    #Tensor_TrainVal = np.lib.stride_tricks.sliding_window_view(Tensor_TrainVal,(sequence_length,Tensor_TrainVal.shape[1]))[:,0,:,:]
    #Tensor_Test = np.lib.stride_tricks.sliding_window_view(Tensor_Test,(sequence_length,Tensor_Test.shape[1]))[:,0,:,:]

    print('Computing representations...')

    length = Tensor_TrainVal_Raw.shape[0]-sequence_length_1+1
    Tensor_TrainVal_Pre = np.zeros(shape=(length,sequence_length_1,Tensor_TrainVal_Raw.shape[1]))
    for n in range(sequence_length_1):
        Tensor_TrainVal_Pre[:,n] = Tensor_TrainVal_Raw[n:length+n]

    length = Tensor_Test_Raw.shape[0]-sequence_length_1+1
    Tensor_Test_Pre = np.zeros(shape=(length,sequence_length_1,Tensor_Test_Raw.shape[1]))
    for n in range(sequence_length_1):
        Tensor_Test_Pre[:,n] = Tensor_Test_Raw[n:length+n]

    Classes_TrainVal_Pre = Classes_TrainVal_Raw[sequence_length_1//2-1:-sequence_length_1//2]
    Classes_Test_Pre = Classes_Test_Raw[sequence_length_1//2-1:-sequence_length_1//2]

    print('Still computing representations...')

    length = Tensor_TrainVal_Pre.shape[0]-sequence_length_2+1
    Tensor_TrainVal = np.zeros(shape=(length,sequence_length_2,sequence_length_1,Tensor_TrainVal_Pre.shape[-1]))
    Classes_TrainVal = np.zeros(shape=(length,sequence_length_2))
    for n in range(sequence_length_2):
        Tensor_TrainVal[:,n] = Tensor_TrainVal_Pre[n:length+n]
        Classes_TrainVal[:,n] = Classes_TrainVal_Pre[n:length+n]
    Tensor_TrainVal = Tensor_TrainVal[::hop]
    Classes_TrainVal = Classes_TrainVal[::hop]

    length = Tensor_Test_Pre.shape[0]-sequence_length_2+1
    Tensor_Test = np.zeros(shape=(length,sequence_length_2,sequence_length_1,Tensor_Test_Pre.shape[-1]))
    Classes_Test = np.zeros(shape=(length,sequence_length_2))
    for n in range(sequence_length_2):
        Tensor_Test[:,n] = Tensor_Test_Pre[n:length+n]
        Classes_Test[:,n] = Classes_Test_Pre[n:length+n]
    Tensor_Test = Tensor_Test[::hop]
    Classes_Test = Classes_Test[::hop]

    Tensor_TrainVal = Tensor_TrainVal.astype(np.float32)
    Classes_TrainVal = Classes_TrainVal.astype(np.float32)
    Tensor_Test = Tensor_Test.astype(np.float32)
    Classes_Test = Classes_Test.astype(np.float32)

    print('Done.')
    print('Normalising representations...')

    Tensor_TrainVal = np.log(Tensor_TrainVal+1e-4)
    min_norm = np.min(Tensor_TrainVal)
    max_norm = np.max(Tensor_TrainVal)
    Tensor_TrainVal = (Tensor_TrainVal-min_norm)/(max_norm-min_norm+1e-16)
    Tensor_Test = np.log(Tensor_Test+1e-4)
    Tensor_Test = (Tensor_Test-min_norm)/(max_norm-min_norm+1e-16)

    print('Done.')

    Tensor_TrainVal_Reduced = np.sum(Tensor_TrainVal, axis=1)
    Classes_TrainVal_Reduced = np.clip(np.sum(Classes_TrainVal, axis=1), 0, 1)
    Tensor_Test_Reduced = np.sum(Tensor_Test, axis=1)
    Classes_Test_Reduced = np.clip(np.sum(Classes_Test, axis=1), 0, 1)

    Dataset_Test = np.expand_dims(Tensor_Test,axis=-1)

    Dataset_Test = Dataset_Test.astype('float32')
    Classes_Test = Classes_Test.astype('float32')

    for it in range(5):

        skf = KFold(n_splits=num_crossval)

        thresholds_crossval = np.zeros(num_crossval)
        pre_avg_crossval = np.zeros(num_crossval)
        post_avg_crossval = np.zeros(num_crossval)
        pre_max_crossval = np.zeros(num_crossval)
        post_max_crossval = np.zeros(num_crossval)

        min_norm_crossval = np.zeros(num_crossval)
        max_norm_crossval = np.zeros(num_crossval)

        pred_norm = []

        validation_accuracy = 0
        test_accuracy = 0

        min_val_loss = np.inf

        set_seeds(0)

        models = []
        g = 0

        for train_index, test_index in skf.split(Tensor_TrainVal_Reduced, Classes_TrainVal_Reduced):

            Dataset_Train, Dataset_Val = Tensor_TrainVal[train_index], Tensor_TrainVal[test_index]
            Classes_Train, Classes_Val = Classes_TrainVal[train_index], Classes_TrainVal[test_index]

            Dataset_Train = np.expand_dims(Dataset_Train,axis=-1).astype('float32')
            Dataset_Val = np.expand_dims(Dataset_Val,axis=-1).astype('float32')

            Classes_Train = Classes_Train.astype('float32')
            Classes_Val = Classes_Val.astype('float32')

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_early, verbose=2, mode='auto', baseline=None, restore_best_weights=True)
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=patience_lr, verbose=2)

            #with tf.device(gpu_name):
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                set_seeds(it)
                model = CRNN_1S_2D(sequence_length_2, dropout)
                set_seeds(it)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)) # , metrics=['accuracy']
                set_seeds(it)
                history = model.fit(Dataset_Train, Classes_Train, batch_size=batch_size, epochs=epochs, validation_data=(Dataset_Val, Classes_Val), callbacks=[early_stopping,lr_scheduler], shuffle=True)

            if min(history.history['val_loss'])<min_val_loss:
                idx_best_model = g
                min_val_loss = min(history.history['val_loss'])

            print('Val Loss for fold ' + str(g+1) + ' of ' + str(num_crossval) + ': ' + str(min(history.history['val_loss'])))

            models.append(model)

            # Calculate threshold parameter with validation set

            print('Processing validation data...')

            #pred_train = flatten_sequence(model.predict(Dataset_Train.astype('float32')),hop)
            #pred_val = flatten_sequence(model.predict(Dataset_Val.astype('float32')),hop)
            #pred_all = np.concatenate((pred_train,pred_val))

            #pred_norm.append([np.max(pred_all),np.min(pred_all)])

            #Classes_Val[Classes_Val!=1] = 0
            #hop_size_ms = hop_size/22050

            #Prediction = (pred_val-pred_norm[g][1])/(pred_norm[g][0]-pred_norm[g][1])
            #Target = flatten_sequence(Classes_Val,hop)

            #predictions = model.predict(Dataset_Val.astype('float32'))

            #Prediction = tf.math.sigmoid(flatten_sequence(predictions, hop))
            #Prediction = flatten_sequence(predictions, hop)
            #Target = flatten_sequence(Classes_Val, hop)

            Classes_TrainVal_CV = Classes_TrainVal.copy()
            Classes_TrainVal_CV[Classes_TrainVal_CV!=1] = 0
            hop_size_ms = hop_size/22050

            predictions_all = model.predict(np.expand_dims(Tensor_TrainVal,axis=-1).astype('float32'))
            Prediction = flatten_sequence(predictions_all, hop)
            Target = flatten_sequence(Classes_TrainVal_CV, hop)

            min_norm_crossval[g] = np.min(Prediction)
            max_norm_crossval[g] = np.max(Prediction)
            Prediction = (Prediction-min_norm_crossval[g])/(max_norm_crossval[g]-min_norm_crossval[g])

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

            f1_score = np.zeros((len(Threshold),len(pre_avg),len(post_avg),len(pre_max),len(post_max)))
            precision = np.zeros((len(Threshold),len(pre_avg),len(post_avg),len(pre_max),len(post_max)))
            recall = np.zeros((len(Threshold),len(pre_avg),len(post_avg),len(pre_max),len(post_max)))
            for i in range(len(Threshold)):
                for c in range(len(pre_avg)):
                    for d in range(len(post_avg)):
                        for e in range(len(pre_max)):
                            for f in range(len(post_max)):
                                pick_picker = madmom.features.onsets.OnsetPeakPickingProcessor(fps=172.265625,pre_avg=pre_avg[c],post_avg=post_avg[d],pre_max=pre_max[e],post_max=post_max[f],threshold=Threshold[i])
                                Pred = pick_picker(Prediction)
                                #Pred = np.argwhere(Prediction>=Threshold[i])[:,0]*hop_size_ms
                                #ind_delete = [i+1 for (x,y,i) in zip(Pred,Pred[1:],range(len(Pred))) if 0.015>abs(x-y)]
                                #Pred = np.delete(Pred, ind_delete)
                                f1_score[i,c,d,e,f], precision[i,c,d,e,f], recall[i,c,d,e,f] = f_measure(Target, Pred, window=0.03)
                print(str(i+1) + '/' + str(len(Threshold)))

            max_f1 = 0
            idx_max = np.zeros(5)
            for i in range(len(Threshold)):
                for c in range(len(pre_avg)):
                    for d in range(len(post_avg)):
                        for e in range(len(pre_max)):
                            for f in range(len(post_max)):
                                if np.mean(f1_score[i,c,d,e,f])>max_f1:
                                    max_f1 = np.mean(f1_score[i,c,d,e,f])
                                    scores_max = f1_score[i,c,d,e,f]
                                    idx_max = np.array([i,c,d,e,f])
            print(idx_max)

            print('Val Accuracy for fold ' + str(g+1) + ' of ' + str(num_crossval) + ': ' + str(np.max(f1_score)))

            thresholds_crossval[g] = Threshold[idx_max[0]]
            pre_avg_crossval[g] = pre_avg[idx_max[1]]
            post_avg_crossval[g] = post_avg[idx_max[2]]
            pre_max_crossval[g] = pre_max[idx_max[3]]
            post_max_crossval[g] = post_max[idx_max[4]]

            g += 1

            tf.keras.backend.clear_session()

            break

        # Test

        print('Evaluating...')

        #idx_model = (np.abs((thresholds_crossval-thresholds_crossval.mean()))).argmin()
        idx_model = 0
        model = models[idx_model]

        pre_avg_eval = pre_avg_crossval[idx_model]
        post_avg_eval = post_avg_crossval[idx_model]
        pre_max_eval = pre_max_crossval[idx_model]
        post_max_eval = post_max_crossval[idx_model]

        predictions = model.predict(Dataset_Test.astype('float32'))

        hop_size_ms = hop_size/22050

        #Prediction = (flatten_sequence(predictions,hop)-pred_norm[idx_best_model][1])/(pred_norm[idx_best_model][0]-pred_norm[idx_best_model][1])
        #Target = flatten_sequence(Classes_Test, hop)

        #Prediction = tf.math.sigmoid(flatten_sequence(predictions, hop))
        Prediction = flatten_sequence(predictions, hop)
        Target = flatten_sequence(Classes_Test, hop)

        factor = np.arange(len(Target))*hop_size_ms
        Target = factor*Target

        j = np.where(Target!=0)
        Target = Target[j]
        
        Target = Target[:Target.argmax()]

        for s in range(len(Target)-1):
            if Target[s+1]<Target[s]:
                print('Ensuring Monotonic Target')
                Target[s+1] = Target[s]

        Prediction = (Prediction-min_norm_crossval[idx_model])/(max_norm_crossval[idx_model]-min_norm_crossval[idx_model])
        threshold = thresholds_crossval[idx_model]

        pick_picker = madmom.features.onsets.OnsetPeakPickingProcessor(fps=172.265625,pre_avg=pre_avg_eval,post_avg=post_avg_eval,pre_max=pre_max_eval,post_max=post_max_eval,threshold=threshold)
        Pred = pick_picker(Prediction)
        #Pred = np.argwhere(Prediction>=threshold)[:,0]*hop_size_ms
        #ind_delete = [i+1 for (x,y,i) in zip(Pred,Pred[1:],range(len(Pred))) if 0.015>abs(x-y)]
        #Pred = np.delete(Pred, ind_delete)
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
            if abs(min_value)<=0.03:
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

        '''if accuracies[a]==np.max(accuracies):
            for g in range(num_crossval):
                models[idx_best_model].save_weights('../../models/' + mode + '/model_dropout_' + str(dropout) + '_crossval_' + str(idx_best_model) + '.h5')'''

        tf.keras.backend.clear_session()