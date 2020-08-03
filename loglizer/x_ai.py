import sys
import math
import numpy as np
import itertools
import bisect
from loglizer import dataloader
import pandas as pd
from loglizer.models import DeepLog
from loglizer.preprocessing import Vectorizer, Iterator


batch_size = 32
hidden_size = 32
num_directions = 2
topk = 5
train_ratio = 0.2
window_size = 10
epoches = 2
num_workers = 2
device = 0

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
    counter = 0
    for i in range(1, 15):#14player:1:15 #52player:1,53
        for p in itertools.combinations((range(0,14)), i):#14player 0:14#54player 0,52
            coalition.append(p)
            counter+=1
            #if i==14:
            if counter%5000==0:
                print("coalition counter:",counter)

    return coalition

def characteristicFunctionDT(coalition,x_test,y_test,x_train,y_train,playerlist,model,cvalue):
    for i in range(0,len(coalition)):
        shap_X_test = np.copy(x_test)
        shap_X_train=np.copy(x_train)
        diff=set(playerlist)^set(coalition[i])
        if(len(diff)!=0):
            shap_X_test[:, tuple(diff)] = np.zeros(np.shape(shap_X_test[:, tuple(diff)]))
            shap_X_train[:, tuple(diff)] = np.zeros(np.shape(shap_X_train[:, tuple(diff)]))

        model.fit(shap_X_train, y_train)
        precision, recall, f1 = model.evaluate(shap_X_test, y_test)
        cvalue.append(precision)
    return cvalue

def characteristicFunctionDL1(coalition,x_train, window_y_train, y_train, x_test, window_y_test, y_test,playerlist,model,cvalue):
    for i in range(0,len(coalition)):
        shap_X_train_window = window_y_train.copy()
        shap_X_test_window=window_y_test.copy()
        diff=set(playerlist)^set(coalition[i])
        if(len(diff)!=0):
            for k in diff:
                shap_X_train_window[shap_X_train_window==np.unique(window_y_train)[k+1]]='#Pad'
              #  shap_X_test_window[shap_X_test_window == np.unique(window_y_train)[k+1]] ='#Pad'


        feature_extractor = Vectorizer()
        train_dataset = feature_extractor.fit_transform(x_train, shap_X_train_window, y_train)
       # test_dataset = feature_extractor.transform(x_test, shap_X_test_window, y_test)

        train_loader = Iterator(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers).iter
        #test_loader = Iterator(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers).iter


        metrics = model.evaluate(train_loader)
        #metrics = model.evaluate(test_loader)
        # print('Train validation:')
        # metrics = model.evaluate(train_loader)

        # print('Test validation:')
        # metrics = model.evaluate(test_loader)
        if i%5==0:
            print(i)
        cvalue.append(metrics["precision"])
    return cvalue

def characteristicFunctionDL(coalition,x_train,y_train,x_test,y_test,playerlist,model,cvalue):
    window_size = 10
    for i in range(0,len(coalition)):
        shap_X_test = np.copy(x_test)
        shap_X_train=np.copy(x_train)
        diff=set(playerlist)^set(coalition[i])
        if(len(diff)!=0):
            shap_X_test[:, tuple(diff)] = np.zeros(np.shape(shap_X_test[:, tuple(diff)]))
            shap_X_train[:, tuple(diff)] = np.zeros(np.shape(shap_X_train[:, tuple(diff)]))

        (shap_X_train,window_y_train,y_train)=dataloader.recover_window(shap_X_train,window_size,y_train)
        (shap_X_test, window_y_test, y_test) = dataloader.recover_window(shap_X_test, window_size, y_test)

        feature_extractor = Vectorizer()
        train_dataset = feature_extractor.transform(shap_X_train, window_y_train, y_train)
        test_dataset = feature_extractor.transform(shap_X_test, window_y_test, y_test)

        train_loader = Iterator(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers).iter
        test_loader = Iterator(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers).iter

        model.fit(train_loader)
        precision, recall, f1 = model.evaluate(test_loader)
        cvalue.append(precision)
    return cvalue