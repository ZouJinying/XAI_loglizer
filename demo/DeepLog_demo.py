#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../')
from loglizer import dataloader
from loglizer.models import DeepLog
from loglizer.preprocessing import Vectorizer, Iterator
from loglizer import  x_ai



batch_size = 32
hidden_size = 32
num_directions = 2
topk = 5
train_ratio = 0.2
window_size = 2#10
epoches = 2
num_workers = 2
device = 0 

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

if __name__ == '__main__':

    Eventname=['E5', 'E22', 'E11', 'E9', 'E26', 'E3', 'E4', 'E2', 'E23', 'E21', 'E20',
       'E25', 'E18', 'E6']
    playerlist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    #playerlist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]
    coalition=[]
    coalition=x_ai.getcoaltionlist()
    cvalue=[]
    print("coalition number:" ,len(coalition))


    (x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = dataloader.load_HDFS(struct_log, label_file=label_file, window='session', window_size=window_size, train_ratio=train_ratio, split_type='uniform')
    
    feature_extractor = Vectorizer()
    train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
    test_dataset = feature_extractor.transform(x_test, window_y_test, y_test)

    train_loader = Iterator(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers).iter
    test_loader = Iterator(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers).iter

    model = DeepLog(num_labels=feature_extractor.num_labels, hidden_size=hidden_size, num_directions=num_directions, topk=topk, device=device)
    model.fit(train_loader, epoches)

   # print('Train validation:')
   # metrics = model.evaluate(train_loader)

   # print('Test validation:')
   # metrics = model.evaluate(test_loader)


    print('calculate characteristic function for coalition')
    x_ai.characteristicFunctionDL1(coalition, x_train, window_y_train, y_train, x_test, window_y_test, y_test, playerlist, model, cvalue)

    print("V1-14:", cvalue[0:14])
    print("V-1:", cvalue[-1])

    print('prepared characteristic value')
    shapleys = x_ai.calculatShapley(cvalue, coalition, playerlist)
    print(shapleys)


    plt.bar(range(14), shapleys, color='lightsteelblue')
    plt.plot(range(14), shapleys, marker='o', color='coral')  # coral
    plt.xticks(range(14), Eventname)
    plt.xlabel('Event')
    plt.ylabel("Shapley addictive index")
    plt.legend()
    plt.show()

