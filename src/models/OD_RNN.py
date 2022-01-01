import os
import pdb
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data as utils

from torch.autograd import Variable

from sklearn.metrics import f1_score
from mir_eval.onset import f_measure

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import optuna


os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.nice(0)


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


class RNN(nn.Module):

    def __init__(self, input_dimensions=4, hidden_dimensions=512, num_classes=2, num_layers=2, dropout=0.0, batch_first=True, bidirectional=True):
        
        super(RNN, self).__init__()
        
        self.input_dimensions = input_dimensions
        self.hidden_dimensions = hidden_dimensions

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout

        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.gru = nn.GRU(input_size=self.input_dimensions, hidden_size=self.hidden_dimensions,
                          num_layers=self.num_layers, batch_first=self.batch_first,
                          dropout=self.dropout, bidirectional=self.bidirectional)

        if self.bidirectional:
            self.fc = nn.Linear(2*self.hidden_dimensions, 1)
        else:
            self.fc = nn.Linear(self.hidden_dimensions, 1)

    def forward(self, x):
        
        assert len(x.size()) == 3, '[GRU]: Input dimension must be of length 3 i.e. [MxSxN]'

        batch_index = 0 if self.batch_first else 1
        num_direction = 2 if self.bidirectional else 1

        h_0 = torch.zeros(self.num_layers*num_direction, x.size(batch_index), self.hidden_dimensions)
        
        self.gru.flatten_parameters()
        output_gru, h_n = self.gru(x, h_0)

        fc_output = self.fc(output_gru)
        fc_output = fc_output.view(-1, fc_output.size(2))
        
        return fc_output

    
class RNN_GPU(nn.Module):

    def __init__(self, input_dimensions=4, hidden_dimensions=512, num_classes=2, num_layers=2, dropout=0.0, batch_first=True, bidirectional=True):
        
        super(RNN_GPU, self).__init__()
        
        self.input_dimensions = input_dimensions
        self.hidden_dimensions = hidden_dimensions

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout

        self.batch_first = batch_first
        self.bidirectional = bidirectional

        fix_seeds(0)
        
        self.gru = nn.GRU(input_size=self.input_dimensions, hidden_size=self.hidden_dimensions,
                          num_layers=self.num_layers, batch_first=self.batch_first,
                          dropout=self.dropout, bidirectional=self.bidirectional)

        fix_seeds(0)
        
        if self.bidirectional:
            self.fc = nn.Linear(2*self.hidden_dimensions, 1)
        else:
            self.fc = nn.Linear(self.hidden_dimensions, 1)

    def forward(self, x):
        
        assert len(x.size()) == 3, '[GRU]: Input dimension must be of length 3 i.e. [MxSxN]'

        batch_index = 0 if self.batch_first else 1
        num_direction = 2 if self.bidirectional else 1

        h_0 = torch.zeros(self.num_layers*num_direction, x.size(batch_index), self.hidden_dimensions).cuda()
        
        fix_seeds(0)
        self.gru.flatten_parameters()
        fix_seeds(0)
        output_gru, h_n = self.gru(x, h_0)
        
        fix_seeds(0)

        fc_output = self.fc(output_gru)
        fix_seeds(0)
        fc_output = fc_output.view(-1, fc_output.size(2))
        
        return fc_output

    
class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_min = 0
        self.delta = delta
    
    def __call__(self, val_loss, model):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score - self.delta:
            self.counter += 1
            #print('EarlyStopping counter: ' + str(self.counter) + ' out of ' + str(self.patience))
            #print('\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        #if self.verbose:
            #print('Validation loss decreased (' + str(self.val_loss_min) + ' --> ' + str(val_loss) + ').  Saving model ...')
        self.val_loss_min = val_loss
        
    
class EarlyStopping_Acc:
    
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta
    
    def __call__(self, val_acc, model):
        
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score <= self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        self.val_acc_max = val_acc
        

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
num_crossval = 10
factor_val = 1.0/num_crossval
factor_train = 1.0 - factor_val

n_epochs = 100
num_classes = 2
sequence_length = 20
num_thresholds_F1_score = 20.
min_sep = 3

patience_lr = 4
patience_early = 7

best_model_save = 'accuracy'
cuda = torch.cuda.is_available()

hop_size = 220

def objective(trial):

    dropout = 0.5
    frame_size = trial.suggest_categorical('frame_size', ['256','512'])
    num_freq = trial.suggest_categorical('num_freq', ['32','64','96'])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    batch_size = 512
    
    AVP_Dataset = np.load('../Data/OD_Datasets/Dataset_AVP_' + num_freq + '_' + frame_size + '.npy')
    BTX_Dataset = np.load('../Data/OD_Datasets/Dataset_BTX_' + num_freq + '_' + frame_size + '.npy')
    FSB_Multi_Dataset = np.load('../Data/OD_Datasets/Dataset_FSB_Multi_' + num_freq + '_' + frame_size + '.npy')
    LVT_2_Dataset = np.load('../Data/OD_Datasets/Dataset_LVT_2_' + num_freq + '_' + frame_size + '.npy')
    LVT_3_Dataset = np.load('../Data/OD_Datasets/Dataset_LVT_3_' + num_freq + '_' + frame_size + '.npy')
    VIM_Dataset = np.load('../Data/OD_Datasets/Dataset_VIM_' + num_freq + '_' + frame_size + '.npy')

    Tensor_All = np.concatenate((AVP_Dataset,BTX_Dataset,FSB_Multi_Dataset,LVT_2_Dataset,LVT_3_Dataset,VIM_Dataset))

    AVP_Labels = np.load('../Data/OD_Datasets/Classes_AVP.npy')
    BTX_Labels = np.load('../Data/OD_Datasets/Classes_BTX.npy')
    FSB_Multi_Labels = np.load('../Data/OD_Datasets/Classes_FSB_Multi.npy')
    LVT_2_Labels = np.load('../Data/OD_Datasets/Classes_LVT_2.npy')
    LVT_3_Labels = np.load('../Data/OD_Datasets/Classes_LVT_3.npy')
    VIM_Labels = np.load('../Data/OD_Datasets/Classes_VIM.npy')

    Labels_All = np.concatenate((AVP_Labels,BTX_Labels,FSB_Multi_Labels,LVT_2_Labels,LVT_3_Labels,VIM_Labels))
    
    for i in range(len(Labels_All)):
        if Labels_All[i]==1:
            Labels_All[i-1]==0.2
            Labels_All[i+1]==0.5
            Labels_All[i+2]==0.1

    fix_seeds(0)

    cut_length = int(Tensor_All.shape[0]/sequence_length)*sequence_length

    Labels_All_0 = Labels_All[:cut_length]
    Tensor_All_0 = Tensor_All[:cut_length]

    factor_div = 5
    div_sequence_length = sequence_length//factor_div

    Tensor_All = np.zeros((int(Tensor_All_0.shape[0]/div_sequence_length)-(factor_div-1), sequence_length, Tensor_All_0.shape[1]))
    Labels_All = np.zeros((int(len(Labels_All_0)/div_sequence_length)-(factor_div-1), sequence_length))

    for n in range(int(Tensor_All_0.shape[0]/div_sequence_length)-(factor_div-1)):
        point = n*div_sequence_length
        Tensor_All[n] = Tensor_All_0[point:point+sequence_length]
        Labels_All[n] = Labels_All_0[point:point+sequence_length]
        
    '''list_delete = []
    for n in range(len(Tensor_All)):
        if np.sum(Labels_All[n])>=1:
            if np.sum(Labels_All[n][-3:])>=1 and np.sum(Labels_All[n][:3])>=1:
                list_delete.append(n)
            else:
                if np.sum(Labels_All[n][-2:])>=1:
                    Tensor_All[n][:-2] = Tensor_All[n][2:]
                    Tensor_All[n][-2:] = Tensor_All[n+1][:2]
                    Labels_All[n][:-2] = Labels_All[n][2:]
                    Labels_All[n][-2:] = Labels_All[n+1][:2]
                elif np.sum(Labels_All[n][:2])>=1:
                    Tensor_All[n][:-2] = Tensor_All[n][2:]'''

    '''count = 0
    list_delete = []   
    for n in range(len(Tensor_All)):
        n += count
        if n>=len(Tensor_All)-1:
            break
        if np.sum(Labels_All[n])>1:
            list_delete.append(n)
        elif np.sum(Labels_All[n])==1:
            if np.sum(Labels_All[n][-5:])>=1 or np.sum(Labels_All[n][:5])>=1:
                list_delete.append(n)
            else:
                list_delete.append(n+1)
                count += 1
        else:
            list_delete.append(n+1)
            count += 1'''
    
    '''count = 0
    list_delete = []   
    for n in range(len(Tensor_All)):
        n += count
        if n>=len(Tensor_All)-1:
            break
        #if np.sum(Labels_All[n])>1:
            #list_delete.append(n)
        if np.sum(Labels_All[n])>=1:
            if np.sum(Labels_All[n][-3:])>=1 or np.sum(Labels_All[n][:3])>=1:
                list_delete.append(n)
            else:
                list_delete.append(n+1)
                count += 1
        else:
            list_delete.append(n+1)
            count += 1

    Tensor_All = np.delete(Tensor_All, list_delete, 0)
    Labels_All = np.delete(Labels_All, list_delete, 0)'''

    std = np.std(Tensor_All)
    mean = np.mean(Tensor_All)
    Tensor_All = (Tensor_All-mean)/std

    #np.random.seed(0)
    #np.random.shuffle(Labels_All)

    #np.random.seed(0)
    #np.random.shuffle(Tensor_All)

    Tensor_All_Reduced = np.sum(Tensor_All, axis=1)
    Labels_All_Reduced = np.sum(Labels_All, axis=1)
    Labels_All_Reduced = np.clip(Labels_All_Reduced, 0, 1)

    loss = torch.nn.BCEWithLogitsLoss()    
    sig = torch.nn.Sigmoid()

    #skf = StratifiedKFold(n_splits=10)
    skf = KFold(n_splits=10)
    g = 0
    
    test_accuracies = np.zeros(num_crossval)
    test_precisions = np.zeros(num_crossval)
    test_recalls = np.zeros(num_crossval)

    for train_index, test_index in skf.split(Tensor_All_Reduced, Labels_All_Reduced):

        Tensor_Train_Val, Tensor_Test = Tensor_All[train_index], Tensor_All[test_index]
        Labels_Train_Val, Labels_Test = Labels_All[train_index], Labels_All[test_index]

        cutoff_index = -(Tensor_Train_Val.shape[0]//9)

        Tensor_Train = Tensor_Train_Val[:cutoff_index]
        Tensor_Val = Tensor_Train_Val[cutoff_index:]

        Labels_Train = Labels_Train_Val[:cutoff_index]
        Labels_Val = Labels_Train_Val[cutoff_index:]

        input_size = Tensor_Train.shape[2]

        if cuda:
            model = RNN_GPU(input_dimensions=input_size, hidden_dimensions=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
            model.cuda()
        else:
            model = RNN(input_dimensions=input_size, hidden_dimensions=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)

        Tensor_Train = torch.from_numpy(Tensor_Train)
        Labels_Train = torch.from_numpy(Labels_Train.astype(int))
        Tensor_Val = torch.from_numpy(Tensor_Val)
        Labels_Val = torch.from_numpy(Labels_Val.astype(int))
        Tensor_Test = torch.from_numpy(Tensor_Test)
        Labels_Test = torch.from_numpy(Labels_Test.astype(int))

        Train_Dataset = utils.TensorDataset(Tensor_Train, Labels_Train)
        Val_Dataset = utils.TensorDataset(Tensor_Val, Labels_Val)
        Test_Dataset = utils.TensorDataset(Tensor_Test, Labels_Test)

        fix_seeds(0)
        Train_Loader = torch.utils.data.DataLoader(Train_Dataset, batch_size=batch_size, shuffle=True)
        fix_seeds(0)
        Val_Loader = torch.utils.data.DataLoader(Val_Dataset, batch_size=Tensor_Val.size()[0]//2+1, shuffle=False)
        fix_seeds(0)
        Test_Loader = torch.utils.data.DataLoader(Test_Dataset, batch_size=Tensor_Test.size()[0]//2+1, shuffle=False)
        fix_seeds(0)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience_lr)
        early_stopping = EarlyStopping_Acc(patience=patience_early, verbose=False)

        best_validation_loss = 1000.
        best_validation_accuracy = 0.

        for epoch in range(n_epochs):

            fix_seeds(0)

            train_loss = 0.

            validation_loss = 0.
            validation_accuracy = 0.

            count_batch_train = 0.

            for batch in Train_Loader:

                data, classes = batch

                data = data.float()
                if cuda:
                    data = data.cuda()
                    classes = classes.cuda()

                #classes = classes.double()

                predictions = model(data)

                predictions = predictions.squeeze(-1)
                predictions = predictions.double()
                predictions = torch.reshape(predictions, (data.size()[0],sequence_length))
                classes = classes.type_as(predictions)
                t_loss = loss(predictions, classes)

                optimizer.zero_grad()
                t_loss.backward()
                optimizer.step()

                train_loss += t_loss.item()

                count_batch_train += 1.

            count_batch_val = 0.

            with torch.no_grad():

                for batch in Val_Loader:

                    data, classes = batch

                    data = data.float()
                    if cuda:
                        data = data.cuda()
                        classes = classes.cuda()

                    #classes = classes.double()

                    predictions = model(data)

                    predictions = predictions.squeeze(-1)
                    predictions = predictions.double()
                    predictions = torch.reshape(predictions, (data.size()[0],sequence_length))
                    classes = classes.type_as(predictions)
                    v_loss = loss(predictions, classes)

                    # Validation_accuracy
                    
                    classes[classes==0.1] = 0
                    classes[classes==0.5] = 0
                    classes[classes==0.2] = 0

                    hop_size_ms = hop_size/44100

                    Prediction = flatten_sequence(sig(predictions).cpu().numpy(), factor_div)
                    Target = flatten_sequence(classes.cpu().numpy(), factor_div)

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
                    
                    optimizer.zero_grad()

                    validation_loss += v_loss.item()
                    validation_accuracy += np.max(f1_score)

                    scheduler.step(validation_loss)

                    count_batch_val += 1.

            train_loss /= float(count_batch_train)
            validation_loss /= float(count_batch_val)
            validation_accuracy /= float(count_batch_val)

            if best_model_save=='loss':
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save(model.state_dict(), 'best_models/OD_RNN')
            elif best_model_save=='accuracy':
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    torch.save(model.state_dict(), 'best_models/OD_RNN')

            print('Train Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(train_loss, validation_loss, validation_accuracy))

            early_stopping(validation_accuracy, model)
            if early_stopping.early_stop or np.isnan(validation_loss):
                print("Early stopping")
                break

        test_accuracy = 0
        test_precision = 0
        test_recall = 0

        if cuda:
            model = RNN_GPU(input_dimensions=input_size, hidden_dimensions=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
            model.cuda()
        else:
            model = RNN(input_dimensions=input_size, hidden_dimensions=hidden_size, num_classes=num_classes, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)

        model.load_state_dict(torch.load('best_models/OD_RNN'))
        model.eval()

        count_batch_test = 0

        with torch.no_grad():

            for batch in Test_Loader:

                data, classes = batch

                data = data.float()
                if cuda:
                    data = data.cuda()
                    classes = classes.cuda()

                #classes = classes.double()

                predictions = model(data)

                predictions = predictions.squeeze(-1)
                predictions = predictions.double()
                predictions = torch.reshape(predictions, (data.size()[0],sequence_length))
                classes = classes.type_as(predictions)

                # Test_accuracy

                '''Target = classes.numpy().flatten()
                Prediction = sig(predictions).numpy().flatten()
                num_thresholds = num_thresholds_F1_score

                Threshold = np.arange(int(num_thresholds+2))/(num_thresholds+2)
                Threshold = Threshold[1:-1]
                F1_Score = np.zeros(len(Threshold))

                scores = np.zeros(len(Threshold))
                for i in range(len(Threshold)):
                    scores[i] = f1_score(Target, Prediction>Threshold[i])
                validation_accuracy = np.max(scores)'''
                
                classes[classes==0.1] = 0
                classes[classes==0.5] = 0
                classes[classes==0.2] = 0

                hop_size_ms = hop_size/44100
                
                Prediction = flatten_sequence(sig(predictions).cpu().numpy(), factor_div)
                Target = flatten_sequence(classes.cpu().numpy(), factor_div)

                factor = np.arange(len(Target))*hop_size_ms
                Target = factor*Target
                
                j = np.where(Target!=0)
                Target = Target[j]
                
                Target = Target[:Target.argmax()]

                num_thresholds = 100
                Threshold = np.arange(int(num_thresholds+2))/(num_thresholds+2)
                Threshold = Threshold[1:-1]

                f1_score = np.zeros(len(Threshold))
                precision = np.zeros(len(Threshold))
                recall = np.zeros(len(Threshold))
                
                f1_score_max = 0
                for i in range(len(Threshold)):
                    Predicted = [1 if item>Threshold[i] else 0 for item in Prediction]
                    Predicted = np.array(Predicted)*factor
                    j = np.where(Predicted!=0)
                    Pred = Predicted[j]
                    ind_delete = [i+1 for (x,y,i) in zip(Pred,Pred[1:],range(len(Pred))) if 0.015>abs(x-y)]
                    Pred = np.delete(Pred, ind_delete)
                    f1_score[i], precision[i], recall[i] = f_measure(Target, Pred, window=0.03)
                    if f1_score[i]>=f1_score_max:
                        Pred_MaxF1 = Pred.copy()
                        f1_score_max = f1_score[i]

                optimizer.zero_grad()

                test_accuracy += np.max(f1_score)
                test_precision += precision[f1_score.argmax()]
                test_recall += recall[f1_score.argmax()]

                count_batch_test += 1.

                if g==0:
                    min_values = []

                min_indices = []
                for k in range(len(Pred_MaxF1)):
                    abs_diff = Target-Pred_MaxF1[k]
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
    
    print('Median Deviation All: ' + str(frame_dev_median))
    print('Mean Deviation All: ' + str(frame_dev_mean))
    print('STD Deviation All: ' + str(frame_dev_std))
    
    print('Mean Accuracy: ' + str(mean_accuracy))
    print('Mean Precision: ' + str(mean_precision))
    print('Mean Recall: ' + str(mean_recall))

    return mean_accuracy

study = optuna.create_study(study_name='OD_RNN', storage='sqlite:///OD_RNN.db', direction='maximize', load_if_exists=True)
study.optimize(objective, n_trials=100)

print(study.best_params)  # Get best parameters for the objective function.
print(study.best_value)  # Get best objective value.
print(study.best_trial)  # Get best trial's information.
print(study.trials)  # Get all trials' information.