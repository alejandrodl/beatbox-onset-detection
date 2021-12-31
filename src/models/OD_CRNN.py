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


os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

    
class CRNN(nn.Module):
    def __init__(self, layers, filters_height, filters_width, dropout, h_dim=2*4, z_dim=32, num_filt=64, h_dim_rnn=512, layers_rnn=2, dropout_rnn=0.0, batch_first=True, bidirectional=True):
        super(CRNN, self).__init__()
        
        # CNN params
        self.z_dim = z_dim
        self.num_filt = num_filt
        h_dim *= num_filt
        
        # RNN params
        self.h_dim_rnn = h_dim_rnn
        self.layers_rnn = layers_rnn
        self.dropout_rnn = dropout_rnn
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        # CNN
        self.conv3_1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.pool3_1 = nn.MaxPool2d(2, 2)
        self.conv3_2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool3_2 = nn.MaxPool2d(2, 2)
        self.conv3_3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3_3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(h_dim, z_dim)
                
        # RNN
        self.gru = nn.GRU(input_size=self.z_dim, hidden_size=self.h_dim_rnn,
                  num_layers=self.layers_rnn, batch_first=self.batch_first,
                  dropout=self.dropout_rnn, bidirectional=self.bidirectional)

        if self.bidirectional:
            self.fc_rnn = nn.Linear(2*self.h_dim_rnn, 1)
        else:
            self.fc_rnn = nn.Linear(self.h_dim_rnn, 1)

    def forward(self, inputs):
        
        # CNN
        output_cnn = torch.zeros((inputs.shape[0], inputs.shape[1], self.z_dim))
        for i in range(inputs.shape[1]):
            input_cnn = inputs[:,i,:,:]
            input_cnn = input_cnn.unsqueeze(1)
            down1 = self.pool3_1(self.conv3_1(input_cnn))
            down2 = self.pool3_2(self.conv3_2(down1))
            down3 = self.pool3_3(self.conv3_3(down2))
            embeddings = down3.view(down3.size()[0],-1)
            output_cnn[:,i] = self.fc(embeddings)
            
        # RNN param
        assert len(output_cnn.size())==3, '[GRU]: Input dimension must be of length 3 i.e. [MxSxN]'
        batch_index = 0 if self.batch_first else 1
        num_direction = 2 if self.bidirectional else 1

        # RNN
        h_0 = torch.zeros(self.layers_rnn*num_direction, output_cnn.size(batch_index), self.h_dim_rnn).cuda()
        output_cnn = output_cnn.float().cuda()
        self.gru.flatten_parameters()
        output_gru, h_n = self.gru(output_cnn.cuda(), h_0)
        fc_output = self.fc_rnn(output_gru)
        output = fc_output.view(-1, fc_output.size(2))

        return output
    
    
class CRNN_Time(nn.Module):
    def __init__(self, layers, filters_height, filters_width, dropout, h_dim=2*4, z_dim=32, num_filt=64, h_dim_rnn=512, layers_rnn=2, dropout_rnn=0.0, batch_first=True, bidirectional=True):
        super(CRNN_Time, self).__init__()
        
        # CNN params
        self.z_dim = z_dim
        self.num_filt = num_filt
        h_dim *= num_filt
        
        # RNN params
        self.h_dim_rnn = h_dim_rnn
        self.layers_rnn = layers_rnn
        self.dropout_rnn = dropout_rnn
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        # CNN
        self.conv3_1 = nn.Conv2d(1, 16, (5,3), 1, (2,1))
        self.pool3_1 = nn.MaxPool2d(2, 2)
        self.conv3_2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool3_2 = nn.MaxPool2d(2, 2)
        self.conv3_3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool3_3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(h_dim, z_dim)
                
        # RNN
        self.gru = nn.GRU(input_size=self.z_dim, hidden_size=self.h_dim_rnn,
                  num_layers=self.layers_rnn, batch_first=self.batch_first,
                  dropout=self.dropout_rnn, bidirectional=self.bidirectional)

        if self.bidirectional:
            self.fc_rnn = nn.Linear(2*self.h_dim_rnn, 1)
        else:
            self.fc_rnn = nn.Linear(self.h_dim_rnn, 1)

    def forward(self, inputs):
        
        # CNN
        output_cnn = torch.zeros((inputs.shape[0], inputs.shape[1], self.z_dim))
        for i in range(inputs.shape[1]):
            input_cnn = inputs[:,i,:,:]
            input_cnn = input_cnn.unsqueeze(1)
            down1 = self.pool3_1(self.conv3_1(input_cnn))
            down2 = self.pool3_2(self.conv3_2(down1))
            down3 = self.pool3_3(self.conv3_3(down2))
            embeddings = down3.view(down3.size()[0],-1)
            output_cnn[:,i] = self.fc(embeddings)
            
        # RNN param
        assert len(output_cnn.size())==3, '[GRU]: Input dimension must be of length 3 i.e. [MxSxN]'
        batch_index = 0 if self.batch_first else 1
        num_direction = 2 if self.bidirectional else 1

        # RNN
        h_0 = torch.zeros(self.layers_rnn*num_direction, output_cnn.size(batch_index), self.h_dim_rnn).cuda()
        output_cnn = output_cnn.float().cuda()
        self.gru.flatten_parameters()
        output_gru, h_n = self.gru(output_cnn.cuda(), h_0)
        fc_output = self.fc_rnn(output_gru)
        output = fc_output.view(-1, fc_output.size(2))

        return output


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
time_length = 8
sequence_length = 12
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

            zp = np.zeros((time_length//2, Tensor_All.shape[-1]))
            Tensor_All_0 = np.concatenate((zp, Tensor_All, zp))

            Tensor_All = np.zeros((Tensor_All_0.shape[0]-time_length, time_length, Tensor_All_0.shape[1]))
            for n in range(Tensor_All.shape[0]):
                Tensor_All[n] = Tensor_All_0[n:n+time_length]

            cut_length = int(Tensor_All.shape[0]/sequence_length)*sequence_length
            
            Labels_All_0 = Labels_All[:cut_length]
            Tensor_All_0 = Tensor_All[:cut_length]

            '''half_sequence_length = sequence_length//2

            Tensor_All = np.zeros((int(Tensor_All_0.shape[0]/half_sequence_length), sequence_length, time_length, Tensor_All_0.shape[2]))
            Labels_All = np.zeros((int(len(Labels_All_0)/half_sequence_length), sequence_length))

            for n in range(int(Tensor_All_0.shape[0]/half_sequence_length)-1):
                point = n*half_sequence_length
                Tensor_All[n] = Tensor_All_0[point:point+sequence_length]
                Labels_All[n] = Labels_All_0[point:point+sequence_length]'''
                
            factor_div = 4
            div_sequence_length = sequence_length//factor_div

            Tensor_All = np.zeros((int(Tensor_All_0.shape[0]/div_sequence_length)-(factor_div-1), sequence_length, time_length, Tensor_All_0.shape[2]))
            Labels_All = np.zeros((int(len(Labels_All_0)/div_sequence_length)-(factor_div-1), sequence_length))

            for n in range(int(Tensor_All_0.shape[0]/div_sequence_length)-(factor_div-1)):
                point = n*div_sequence_length
                Tensor_All[n] = Tensor_All_0[point:point+sequence_length]
                Labels_All[n] = Labels_All_0[point:point+sequence_length]
                
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

                if net_type=='Standard':
                    model = CRNN(layers=[1,1,1], filters_height=[3,3,3], filters_width=[3,3,3], dropout=0.2, h_dim=1*4, z_dim=32, h_dim_rnn=32, layers_rnn=3, dropout_rnn=0.2, batch_first=True, bidirectional=True).cuda()
                else:
                    model = CRNN_Time(layers=[1,1,1], filters_height=[3,3,3], filters_width=[3,3,3], dropout=0.2, h_dim=1*4, z_dim=32, h_dim_rnn=32, layers_rnn=3, dropout_rnn=0.2, batch_first=True, bidirectional=True).cuda()

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
                            classes = classes.cuda()
                            data = data.cuda()

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
                                classes = classes.cuda()
                                data = data.cuda()

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

                            #print(Target[:100])
                            #print(Prediction[:100])

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
                        if validation_loss <= best_validation_loss:
                            best_validation_loss = validation_loss
                            torch.save(model.state_dict(), 'best_models/OD_CRNN')
                    elif best_model_save=='accuracy':
                        if validation_accuracy >= best_validation_accuracy:
                            best_validation_accuracy = validation_accuracy
                            torch.save(model.state_dict(), 'best_models/OD_CRNN')

                    print('Train Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(train_loss, validation_loss, validation_accuracy))

                    early_stopping(validation_accuracy, model)
                    if early_stopping.early_stop or np.isnan(validation_loss):
                        print("Early stopping")
                        break

                test_accuracy = 0
                test_precision = 0
                test_recall = 0

                if net_type=='Standard':
                    model = CRNN(layers=[1,1,1], filters_height=[3,3,3], filters_width=[3,3,3], dropout=0.2, h_dim=1*4, z_dim=32, h_dim_rnn=32, layers_rnn=3, dropout_rnn=0.2, batch_first=True, bidirectional=True).cuda()
                else:
                    model = CRNN_Time(layers=[1,1,1], filters_height=[3,3,3], filters_width=[3,3,3], dropout=0.2, h_dim=1*4, z_dim=32, h_dim_rnn=32, layers_rnn=3, dropout_rnn=0.2, batch_first=True, bidirectional=True).cuda()

                model.load_state_dict(torch.load('best_models/OD_CRNN'))
                model.eval()

                count_batch_test = 0

                with torch.no_grad():

                    for batch in Test_Loader:

                        data, classes = batch

                        data = data.float()
                        if cuda:
                            classes = classes.cuda()
                            data = data.cuda()

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

            print('LR: ' + str(lr) + ', Frame: ' + frame_size + ', Net: ' + net_type + ', Accuracy: ' + str(mean_accuracy))
