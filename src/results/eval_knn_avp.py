import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import xgboost as xgb
import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from warnings import simplefilter
simplefilter(action='ignore')



# Global parameters

#modes = ['eng_all','eng_mfcc_env','syllall']
modes = ['eng_mfcc_env']
clfs = ['knn3','knn5','knn7','knn9','knn11','logr','rf','xgboost']

class_dict_avp = {'kd':0, 'sd':1, 'hhc':2, 'hho':3}
class_dict_avp_replacer = class_dict_avp.get

percentage_train = 80

num_crossval = 5
num_crosstest = 7
num_iterations = 5

num_iterations_algorithms = 5

# Data parameters

frame_size = '1024'

# Network parameters

latent_dim = 32

# Training parameters

epochs = 10000
batch_size = 128

# Spectrogram normalisation values

norm_min_max = [[0.0, 3.7073483668036347],[-9.210340371976182, 9.999500033329732e-05]]

# KNN parameters

num_neighborss = [3,5,7,9,11]

# Gradient Boosting Trees parameters

max_depth = 7
min_child_weight = 6
colsample = 0.85
subsample = 0.85
learning_rate = 0.03
reg_lambda = 1.0
reg_alpha = 0
n_estimators = 1000

# Placeholder

for mode in modes:
    if not os.path.isdir('results/' + mode):
        os.mkdir('results/' + mode)

    if mode=='eng_mfcc_env':
        num_crossval = 1
        num_iterations = 1
    elif mode=='eng_all':
        num_crossval = 1
        num_iterations = 5
    elif mode=='syllall':
        num_crossval = 5
        num_iterations = 5

    accuracies = np.zeros((num_iterations,num_crossval,len(num_neighborss)+3,28))

    for it_mod in range(num_iterations):
        list_test_participants_avp_all = np.load('data/processed/' + mode + '/list_test_participants_avp_all_' + str(it_mod) + '.npy')

        for ct in range(num_crosstest):
            for cv in range(num_crossval):
                list_test_participants = list_test_participants_avp_all[ct,cv].astype(int)
                
                if mode=='eng_all':
                    kf = KFold(n_splits=5)

                for part in list_test_participants:
                    classes_str = np.load('data/interim/AVP/Classes_Test_' + str(part).zfill(2) + '.npy')
                    classes_eval = [class_dict_avp_replacer(n,n) for n in classes_str]
                    classes_str = np.load('data/interim/AVP/Classes_Train_Aug_' + str(part).zfill(2) + '.npy')
                    classes = [class_dict_avp_replacer(n,n) for n in classes_str]

                    if mode=='syllall':
                        dataset = np.load('data/processed/' + mode + '/train_features_aug_avp_' + mode + '_' + str(part).zfill(2) + '_' + str(ct) + '_' + str(it_mod) + '_' + str(cv) + '.npy').astype('float32')
                        dataset_eval = np.load('data/processed/' + mode + '/test_features_avp_' + mode + '_' + str(part).zfill(2) + '_' + str(ct) + '_' + str(it_mod) + '_' + str(cv) + '.npy').astype('float32')
                    elif mode=='eng_mfcc_env':
                        dataset = np.load('data/processed/' + mode + '/train_features_avp_' + mode + '_32_' + str(part).zfill(2) + '.npy').astype('float32')
                        dataset_eval = np.load('data/processed/' + mode + '/test_features_avp_' + mode + '_32_' + str(part).zfill(2) + '.npy').astype('float32')
                    elif mode=='eng_all':
                        dataset = np.load('data/processed/' + mode + '/train_features_avp_' + mode + '_' + str(part).zfill(2) + '.npy').astype('float32')
                        dataset_eval = np.load('data/processed/' + mode + '/test_features_avp_' + mode + '_' + str(part).zfill(2) + '.npy').astype('float32')

                    classes = np.array(classes).astype('float32')
                    classes_eval = np.array(classes_eval).astype('float32')
                    
                    # Normalisation

                    for feat in range(dataset.shape[-1]):
                        mean = np.mean(np.concatenate((dataset[:,feat],dataset_eval[:,feat])))
                        std = np.std(np.concatenate((dataset[:,feat],dataset_eval[:,feat])))
                        dataset[:,feat] = (dataset[:,feat]-mean)/(std+1e-16)
                        dataset_eval[:,feat] = (dataset_eval[:,feat]-mean)/(std+1e-16)

                    mean = np.mean(np.vstack((dataset,dataset_eval)))
                    std = np.std(np.vstack((dataset,dataset_eval)))
                    dataset = (dataset-mean)/(std+1e-16)
                    dataset_eval = (dataset_eval-mean)/(std+1e-16)

                    np.random.seed(0)
                    np.random.shuffle(dataset)

                    np.random.seed(0)
                    np.random.shuffle(classes)

                    if mode=='eng_all':
                        '''counter = 0
                        for train_index, test_index in kf.split(classes):
                            dataset_train, dataset_val = dataset[train_index], dataset[test_index]
                            classes_train, classes_val = classes[train_index], classes[test_index]
                            if counter==cv:
                                break
                        forest = RandomForestClassifier(n_estimators=5,random_state=it_mod,n_jobs=-1)
                        forest.fit(dataset, classes)
                        results = permutation_importance(forest, dataset_val, classes_val, n_repeats=2, random_state=it_mod)
                        indices_sorted = np.array(results.importances_mean).argsort()[::-1].tolist()
                        
                        dataset = dataset[:,indices_sorted[:32]]
                        dataset_eval = dataset_eval[:,indices_sorted[:32]]'''

                        datasets = np.vstack((dataset,dataset_eval))

                        pca = PCA(n_components=32, svd_solver='randomized', random_state=it_mod)
                        datasets = pca.fit_transform(datasets)

                        dataset = datasets[:dataset.shape[0]]
                        dataset_eval = datasets[dataset.shape[0]:]

                        for feat in range(dataset.shape[-1]):
                            mean = np.mean(np.concatenate((dataset[:,feat],dataset_eval[:,feat])))
                            std = np.std(np.concatenate((dataset[:,feat],dataset_eval[:,feat])))
                            dataset[:,feat] = (dataset[:,feat]-mean)/(std+1e-16)
                            dataset_eval[:,feat] = (dataset_eval[:,feat]-mean)/(std+1e-16)

                        mean = np.mean(np.vstack((dataset,dataset_eval)))
                        std = np.std(np.vstack((dataset,dataset_eval)))
                        dataset = (dataset-mean)/(std+1e-16)
                        dataset_eval = (dataset_eval-mean)/(std+1e-16)

                    for neigh in range(len(num_neighborss)):
                        classifier = KNeighborsClassifier(n_neighbors=num_neighborss[neigh])
                        classifier.fit(dataset, classes)
                        accuracies[it_mod,cv,neigh,part] = classifier.score(dataset_eval, classes_eval)
                    accur = 0
                    for it_alg in range(num_iterations_algorithms):
                        clf = LogisticRegression(solver='liblinear')
                        clf.fit(dataset, classes)
                        accur += clf.score(dataset_eval, classes_eval)
                    accuracies[it_mod,cv,neigh+1,part] = accur/num_iterations_algorithms
                    accur = 0
                    for it_alg in range(num_iterations_algorithms):
                        clf = RandomForestClassifier(random_state=it_alg)
                        clf.fit(dataset, classes)
                        accur += clf.score(dataset_eval, classes_eval)
                    accuracies[it_mod,cv,neigh+2,part] = accur/num_iterations_algorithms
                    accur = 0
                    for it_alg in range(num_iterations_algorithms):
                        params = {'max_depth':max_depth,
                                    'min_child_weight': min_child_weight,
                                    'learning_rate':learning_rate,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample,
                                    'objective':'multi:softmax',
                                    'reg_lambda':reg_lambda,
                                    'reg_alpha':reg_alpha,
                                    'n_estimators':n_estimators,
                                    'random_state':it_alg}
                        model = xgb.XGBClassifier(**params)
                        model.fit(dataset, classes, eval_metric='merror')
                        accur += accuracy_score(classes_eval, model.predict(dataset_eval))
                    accuracies[it_mod,cv,neigh+3,part] = accur/num_iterations_algorithms

    np.save('results/' + mode + '/accuracies', accuracies)
                        