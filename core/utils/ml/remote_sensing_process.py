#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
from sklearn import metrics
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as XGBR
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams


def train():
    all_csv = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_test.csv'
    tool1 = pd.read_csv(all_csv)
    a_samples = tool1.iloc[:, :-1]
    a_labels = tool1.iloc[:, -1]
    print(a_samples)
    print(a_labels)
    a_samples = np.array(a_samples)
    a_samples[np.isnan(a_samples)] = 0
    print(a_samples)
    print(a_labels)
    classifier = RandomForestClassifier(n_jobs=-1)
    # classifier = XGBR(n_jobs=-1)
    classifier.fit(a_samples, a_labels)
    # 通过交叉验证评估分数 cv=50 0.9204926108374386;cv=100 0.0.9197142857142859;cv=200 0.92
    result = CVS(classifier, a_samples, a_labels, cv=10).mean()
    print(result)

    predict_csv = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_all_data_.csv'
    predict_csv_to1 = pd.read_csv(predict_csv)
    rows, columns = predict_csv_to1.shape[0], predict_csv_to1.shape[1]
    predict_csv_to1 = predict_csv_to1.iloc[:, :]
    predict_csv_to1[np.isinf(predict_csv_to1)] = 10
    predict_csv_to1[np.isnan(predict_csv_to1)] = 0
    result = classifier.predict(predict_csv_to1)
    # classification = result.reshape((rows, columns))
    f = plt.figure()
    rcParams['figure.figsize'] = 100, 300

    predict_image = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_map.tif'
    bands_data = []
    raster_dataset = gdal.Open(predict_image, gdal.GA_ReadOnly)
    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()

    band = raster_dataset.GetRasterBand(1)
    bands_data.append(band.ReadAsArray())

    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape

    r = bands_data[:, :, 2]
    g = bands_data[:, :, 1]
    b = bands_data[:, :, 0]
    rgb = np.dstack([r, g, b])
    f.add_subplot(1, 2, 1)
    plt.imshow(rgb / 255)
    f.add_subplot(1, 2, 2)
    # plt.imshow(classification, cmap='gray')


if __name__ == '__main__':
    train()
