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


os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.nice(0)


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


class CNN_Onset_Timbre(nn.Module):
    def __init__(self):
        super(CNN_Onset_Timbre, self).__init__()
        
        self.conv1_1 = nn.Conv2d(1, 8, (3,17), 1, (1,0))
        self.pool1_1 = nn.MaxPool2d((4,2), (4,2))
        self.conv1_2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.pool1_2 = nn.MaxPool2d(2, 2)
        self.conv1_3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool1_3 = nn.MaxPool2d(2, 2)
        
        self.conv2_1 = nn.Conv2d(1, 8, (3,9), 1, (1,0))
        self.pool2_1 = nn.MaxPool2d((4,3), (4,3))
        self.conv2_2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.pool2_2 = nn.MaxPool2d(2, 2)
        self.conv2_3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2_3 = nn.MaxPool2d(2, 2)
        
        self.conv3_1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.pool3_1 = nn.MaxPool2d((4,2), (4,2))
        self.conv3_2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool3_2 = nn.MaxPool2d(2, 2)
        self.conv3_3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3_3 = nn.MaxPool2d(2, 2)
        
        self.conv4_1 = nn.Conv2d(1, 16, (9,3), 1, (0,1))
        self.pool4_1 = nn.MaxPool2d(2, 2)
        self.conv4_2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool4_2 = nn.MaxPool2d(2, 2)
        self.conv4_3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool4_3 = nn.MaxPool2d(2, 2)
        
        self.conv5_1 = nn.Conv2d(1, 32, (13,3), 1, (0,1))
        self.pool5_1 = nn.MaxPool2d((2,4), (2,4))
        self.conv5_2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool5_2 = nn.MaxPool2d(2, 2)
        
        self.out_size = (64*4*3+32*2*2)
        
        self.fc_1 = nn.Linear(self.out_size, 64)
        self.fc_2 = nn.Linear(64, 1)

    def forward(self, x):

        x1 = self.pool1_1(F.relu(self.conv1_1(x)))
        x1 = self.pool1_2(F.relu(self.conv1_2(x1)))
        x1 = self.pool1_3(F.relu(self.conv1_3(x1)))

        x2 = self.pool2_1(F.relu(self.conv2_1(x)))
        x2 = self.pool2_2(F.relu(self.conv2_2(x2)))
        x2 = self.pool2_3(F.relu(self.conv2_3(x2)))
        
        x3 = self.pool3_1(F.relu(self.conv3_1(x)))
        x3 = self.pool3_2(F.relu(self.conv3_2(x3)))
        x3 = self.pool3_3(F.relu(self.conv3_3(x3)))
        
        x4 = self.pool4_1(F.relu(self.conv4_1(x)))
        x4 = self.pool4_2(F.relu(self.conv4_2(x4)))
        x4 = self.pool4_3(F.relu(self.conv4_3(x4)))
        
        x5 = self.pool5_1(F.relu(self.conv5_1(x)))
        x5 = self.pool5_2(F.relu(self.conv5_2(x5)))
        
        x1 = x1.view(x1.size()[0],-1)
        x2 = x2.view(x2.size()[0],-1)
        x3 = x3.view(x3.size()[0],-1)
        x4 = x4.view(x4.size()[0],-1)
        x5 = x5.view(x5.size()[0],-1)
        
        x = torch.cat((x1,x2,x3,x4,x5),dim=1)
        
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        
        return torch.flatten(x)
    
    
class CNN_Onset_Time(nn.Module):
    def __init__(self):
        super(CNN_Onset_Time, self).__init__()
        
        self.conv3_1 = nn.Conv2d(1, 32, (7,3), 1, (3,1))
        self.pool3_1 = nn.MaxPool2d(2, 2)
        self.conv3_2 = nn.Conv2d(32, 64, (5,3), 1, (2,1))
        self.pool3_2 = nn.MaxPool2d(2, 2)
        self.conv3_3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool3_3 = nn.MaxPool2d(2, 2)
        
        self.out_size = 128*8
        
        self.fc_1 = nn.Linear(self.out_size, 64)
        self.fc_2 = nn.Linear(64, 1)

    def forward(self, x):
        
        x = self.pool3_1(F.relu(self.conv3_1(x)))
        x = self.pool3_2(F.relu(self.conv3_2(x)))
        x = self.pool3_3(F.relu(self.conv3_3(x)))
        
        x = x.view(x.size()[0],-1)
        
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        
        return torch.flatten(x)
    
    
class CNN_Onset_Standard(nn.Module):
    def __init__(self):
        super(CNN_Onset_Standard, self).__init__()
        
        self.conv3_1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.pool3_1 = nn.MaxPool2d(2, 2)
        self.conv3_2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3_2 = nn.MaxPool2d(2, 2)
        self.conv3_3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool3_3 = nn.MaxPool2d(2, 2)
        
        self.out_size = 128*8
        
        self.fc_1 = nn.Linear(self.out_size, 64)
        self.fc_2 = nn.Linear(64, 1)

    def forward(self, x):
        
        x = self.pool3_1(F.relu(self.conv3_1(x)))
        x = self.pool3_2(F.relu(self.conv3_2(x)))
        x = self.pool3_3(F.relu(self.conv3_3(x)))
        
        x = x.view(x.size()[0],-1)
        
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        
        return torch.flatten(x)
    

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

test_accuracies = np.zeros(num_crossval)

n_epochs = 100
num_classes = 2
time_length = 16
num_thresholds_F1_score = 100.
min_sep = 3

patience_lr = 4
patience_early = 7

best_model_save = 'accuracy'
cuda = torch.cuda.is_available()

hop_size = 220
batch_size = 1024

num_freq = '32'
            
lrs = [2*1e-4]
frame_sizes = ['2048','1024','512','256']
net_types = ['Time','Standard']

for a in range(len(lrs)):
    
    for b in range(len(frame_sizes)):
        
        for c in range(len(net_types)):
    
            lr = lrs[a]
            frame_size = frame_sizes[b]
            net_type = net_types[c]

            print('LR: ' + str(lr) + ', Frame: ' + frame_size + ', Net: ' + net_type)

            AVP_Dataset = np.load('../Data/OD_Datasets/Dataset_AVP_' + num_freq + '_' + frame_size + '.npy')
            BTX_Dataset = np.load('../Data/OD_Datasets/Dataset_BTX_' + num_freq + '_' + frame_size + '.npy')
            FSB_Multi_Dataset = np.load('../Data/OD_Datasets/Dataset_FSB_Multi_' + num_freq + '_' + frame_size + '.npy')
            LVT_2_Dataset = np.load('../Data/OD_Datasets/Dataset_LVT_2_' + num_freq + '_' + frame_size + '.npy')
            LVT_3_Dataset = np.load('../Data/OD_Datasets/Dataset_LVT_3_' + num_freq + '_' + frame_size + '.npy')
            VIM_Dataset = np.load('../Data/OD_Datasets/Dataset_VIM_' + num_freq + '_' + frame_size + '.npy')

            Tensor_All = np.concatenate((AVP_Dataset,BTX_Dataset,LVT_2_Dataset,VIM_Dataset,FSB_Multi_Dataset,LVT_3_Dataset))

            AVP_Labels = np.load('../Data/OD_Datasets/Classes_AVP.npy')
            BTX_Labels = np.load('../Data/OD_Datasets/Classes_BTX.npy')
            FSB_Multi_Labels = np.load('../Data/OD_Datasets/Classes_FSB_Multi.npy')
            LVT_2_Labels = np.load('../Data/OD_Datasets/Classes_LVT_2.npy')
            LVT_3_Labels = np.load('../Data/OD_Datasets/Classes_LVT_3.npy')
            VIM_Labels = np.load('../Data/OD_Datasets/Classes_VIM.npy')

            Labels_All = np.concatenate((AVP_Labels,BTX_Labels,LVT_2_Labels,VIM_Labels,FSB_Multi_Labels,LVT_3_Labels))
            
            for i in range(len(Labels_All)):
                if Labels_All[i]==1:
                    Labels_All[i-1]==0.2
                    Labels_All[i+1]==0.5
                    Labels_All[i+2]==0.1

            fix_seeds(0)

            zp = np.zeros((time_length//2, Tensor_All.shape[-1]))
            Tensor_All_0 = np.concatenate((zp, Tensor_All, zp))

            Tensor_All = np.zeros((Tensor_All_0.shape[0]-time_length, time_length, Tensor_All_0.shape[1]))
            for n in range(Tensor_All.shape[0]):
                Tensor_All[n] = Tensor_All_0[n:n+time_length]

            std = np.std(Tensor_All)
            mean = np.mean(Tensor_All)
            Tensor_All = (Tensor_All-mean)/std

            Tensor_All = np.expand_dims(Tensor_All,1)

            loss = torch.nn.BCEWithLogitsLoss()    
            sig = torch.nn.Sigmoid()

            Tensor_All, Labels_All = Make_KFolds_CNN(Tensor_All, Labels_All, num_crossval)

            test_accuracies = np.zeros(num_crossval)
            test_precisions = np.zeros(num_crossval)
            test_recalls = np.zeros(num_crossval)

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

                cutoff_index = -(Tensor_Train_Val.shape[0]//9)

                Tensor_Train = Tensor_Train_Val[:cutoff_index]
                Tensor_Val = Tensor_Train_Val[cutoff_index:]

                Labels_Train = Labels_Train_Val[:cutoff_index]
                Labels_Val = Labels_Train_Val[cutoff_index:]

                input_size = Tensor_Train.shape[2]

                if net_type=='Standard':
                    model = CNN_Onset_Standard().cuda()
                else:
                    model = CNN_Onset_Time().cuda()

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

                        predictions = model(data)

                        predictions = predictions.squeeze(-1)
                        predictions = predictions.double()

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

                            predictions = model(data)

                            predictions = predictions.squeeze(-1)
                            predictions = predictions.double()

                            classes = classes.type_as(predictions)
                            v_loss = loss(predictions, classes)

                            # Validation_accuracy

                            classes[classes==0.1] = 0
                            classes[classes==0.5] = 0
                            classes[classes==0.2] = 0

                            hop_size_ms = hop_size/44100

                            if cuda:
                                factor = (np.arange(len(classes.cpu().numpy().flatten()))+1)*hop_size_ms
                                Target = classes.cpu().numpy().flatten()*factor
                                Prediction = sig(predictions).cpu().numpy().flatten()
                            else:
                                factor = (np.arange(len(classes.numpy().flatten()))+1)*hop_size_ms
                                Target = classes.numpy().flatten()*factor
                                Prediction = sig(predictions).numpy().flatten()

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
                            torch.save(model.state_dict(), 'best_models/OD_CNN')
                    elif best_model_save=='accuracy':
                        if validation_accuracy > best_validation_accuracy:
                            best_validation_accuracy = validation_accuracy
                            torch.save(model.state_dict(), 'best_models/OD_CNN')

                    print('Train Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(train_loss, validation_loss, validation_accuracy))

                    early_stopping(validation_accuracy, model)
                    if early_stopping.early_stop or np.isnan(validation_loss):
                        print("Early stopping")
                        break

                test_accuracy = 0
                test_precision = 0
                test_recall = 0

                if net_type=='Standard':
                    model = CNN_Onset_Standard().cuda()
                else:
                    model = CNN_Onset_Time().cuda()

                model.load_state_dict(torch.load('best_models/OD_CNN'))
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

                        if cuda:
                            factor = (np.arange(len(classes.cpu().numpy().flatten()))+1)*hop_size_ms
                            Target = classes.cpu().numpy().flatten()*factor
                            Prediction = sig(predictions).cpu().numpy().flatten()
                        else:
                            factor = (np.arange(len(classes.numpy().flatten()))+1)*hop_size_ms
                            Target = classes.numpy().flatten()*factor
                            Prediction = sig(predictions).numpy().flatten()

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

            print('LR: ' + str(lr) + ', Frame: ' + frame_size + ', Net: ' + net_type + ', Accuracy: ' + str(mean_accuracy))
