#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import SVM
from loglizer import dataloader, preprocessing
from loglizer import x_ai
import matplotlib.pyplot as plt


struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

if __name__ == '__main__':

    Eventname=['E5', 'E22', 'E11', 'E9', 'E26', 'E3', 'E4', 'E2', 'E23', 'E21', 'E20',
       'E25', 'E18', 'E6']
    playerlist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    coalition=x_ai.getcoaltionlist()
    cvalue=[]
    print("coalition number:" ,len(coalition))


    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test,1)

    model = SVM()
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)

    print('calculate characteristic function for coalition')
    x_ai.characteristicFunctionDT(coalition, x_test, y_test, x_train, y_train, playerlist, model, cvalue)

    print("V1-14:",cvalue[0:14])
    print("V-1:",cvalue[-1])

    print('prepared characteristic value')
    shapleys=x_ai.calculatShapley(cvalue,coalition,playerlist)
    print(shapleys)

    plt.bar(range(14), shapleys, color='lightsteelblue')
    plt.plot(range(14), shapleys, marker='o', color='coral')  # coral
    plt.xticks(range(14), Eventname)
    plt.xlabel('Event')
    plt.ylabel("Shapley addictive index")
    plt.legend()
    plt.show()