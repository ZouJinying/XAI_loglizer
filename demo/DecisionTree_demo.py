#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import itertools
import bisect
import matplotlib.pyplot as plt

sys.path.append('../')

from loglizer.models import DecisionTree
from loglizer import dataloader, preprocessing

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file
struct_log_real1 = '../data/HDFS/HDFS.npz'
def calculatShapley(cFunction,coalition,nPlayer):
    coalition=list(coalition)
    for i in range(0,len(coalition)):
        coalition[i]=list(coalition[i])


    print("start calculate shapley:")
    shapley_values = []
    for i in range(len(nPlayer)):
        shapley = 0
        for j in coalition:
            if i not in j:
                j=list(j)
                cmod = len(j)
                Cui = j[:]
                bisect.insort_left(Cui,i)
                l = coalition.index(j)
                k = coalition.index(Cui)
                temp = float(float(cFunction[k]) - float(cFunction[l])) *\
                           float(math.factorial(cmod) * math.factorial(len(nPlayer) - cmod - 1)) / float(math.factorial(len(nPlayer)))
                shapley += temp
                # if i is 0:
                #     print j, Cui, cmod, n-cmod-1, characteristic_function[k], characteristic_function[l], math.factorial(cmod), math.factorial(n - cmod - 1), math.factorial(n)

        cmod = 0
        Cui = [i]
        k = coalition.index(Cui)
        temp = float(cFunction[k]) * float(math.factorial(cmod) * math.factorial(len(nPlayer) - cmod - 1)) / float(math.factorial(len(nPlayer)))
        shapley += temp

        shapley_values.append(shapley)

    return (shapley_values)

def getcoaltionlist():
    coalition=[]
    for i in range(1, 15):
        for p in itertools.combinations((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28), i):
            coalition.append(p)
            #if i==14:

    return coalition

if __name__ == '__main__':




    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                               label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test,1)

    model = DecisionTree()
    model.fit(x_train, y_train)

    shap_X_test = x_test

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)

    # print('Test shapley validation:')
    # for k in range(0,np.shape(x_test)[1]):
    #     shap_X_test = np.copy(x_test)
    #     shap_X_test[:,k] = np.zeros(np.shape(shap_X_test[:,k]))
    #     precision, recall, f1 = model.evaluate(shap_X_test, y_test)

    Eventname=['E5', 'E22', 'E11', 'E9', 'E26', 'E3', 'E4', 'E2', 'E23', 'E21', 'E20',
       'E25', 'E18', 'E6']
    all=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    coalition=getcoaltionlist()
    cvalue=[]
    print("coalition number:" ,len(coalition))

    print('calculate characteristic function for coalition')
    for i in range(0,len(coalition)):
        shap_X_test = np.copy(x_test)
        shap_X_train=np.copy(x_train)
        diff=set(all)^set(coalition[i])
        if(len(diff)!=0):

            shap_X_test[:, tuple(diff)] = np.zeros(np.shape(shap_X_test[:, tuple(diff)]))
            shap_X_train[:, tuple(diff)] = np.zeros(np.shape(shap_X_train[:, tuple(diff)]))

        model.fit(shap_X_train, y_train)
        precision, recall, f1 = model.evaluate(shap_X_test, y_test)

        cvalue.append(precision)

    print("V1-14:",cvalue[0:14])
    print("V-1:",cvalue[-1])

    print('prepared characteristic value')
    shapleys=calculatShapley(cvalue,coalition,all)
    print(shapleys)

    plt.bar(range(29), shapleys, color='lightsteelblue')
    plt.plot(range(29), shapleys, marker='o', color='coral')  # coral
    plt.xticks(range(29), Eventname)
    plt.xlabel('Event')
    plt.ylabel("Shapley addictive index")
    plt.legend()
    plt.show()