#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import PCA
from loglizer import dataloader, preprocessing
# import numpy as np
# np.set_printoptions(threshold=np.inf)

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')
    # print("load_HDFS后的x_train",x_train,type(x_train))
    print("y_train!!!!!!!!:",y_train)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    print("fit_transform后的x_train 尺寸是：",x_train.shape)
    x_test = feature_extractor.transform(x_test)
    # print("输入后的测试数据矩阵:",x_test)

    model = PCA()
    model.fit(x_train)
    print('Train validation:')

    precision, recall, f1 = model.evaluate(x_train, y_train)
    # print("训练出来的y是这个样子：", y_train)
    # print('Test validation:')
    # precision, recall, f1 = model.evaluate(x_test, y_test)
