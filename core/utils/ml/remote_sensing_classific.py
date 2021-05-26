#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import os
from osgeo import gdal
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as XGBR

# A list of "random" colors (for a nicer output)
COLORS = ["#000000", "#DDDD00", "#1CE6FF", "#EE34FF", "#EE4A46", "#00EA41"]
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS


# In[2]:


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    print(vector_data_path)
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize the vectors in the given directory in a single image."""
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i + 1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    return labeled_pixels


def write_geotiff(fname, data, geo_transform, projection):
    """Create a GeoTIFF file with the given data."""
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None  # Close the file


# In[65]:

def tif(raster_data_path_1, raster_data_path_2, raster_data_path_3, raster_data_path_4, all_data_csv_file, shapefiles,
        shape_files_csv_file, output_fname):
    bands_data = []
    '''
    raster_data_path_1 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/xinjiang_shawan_b2.tif"
    raster_data_path_2 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/xinjiang_shawan_b3.tif"
    raster_data_path_3 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/xinjiang_shawan_b4.tif"
    raster_data_path_4 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/xinjiang_shawan_b5.tif"
    output_fname = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/classification.tiff"
    '''
    data_list = [raster_data_path_1, raster_data_path_2, raster_data_path_3, raster_data_path_4]
    for i in range(len(data_list)):
        raster_dataset = gdal.Open(data_list[i], gdal.GA_ReadOnly)
        geo_transform = raster_dataset.GetGeoTransform()
        proj = raster_dataset.GetProjectionRef()

        band = raster_dataset.GetRasterBand(1)
        bands_data.append(band.ReadAsArray())

    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape

    # raster_dataset = gdal.Open(raster_data_path_1, gdal.GA_ReadOnly)
    # geo_transform = raster_dataset.GetGeoTransform()
    # proj = raster_dataset.GetProjectionRef()
    # # bands_data = []
    # print("波段数")
    # print(raster_dataset.RasterCount)

    # for b in range(1, raster_dataset.RasterCount+1):
    #     band = raster_dataset.GetRasterBand(b)
    #     bands_data.append(band.ReadAsArray())
    #
    # bands_data = np.dstack(bands_data)
    # rows, cols, n_bands = bands_data.shape

    print(bands_data.shape)

    flat_pixels = bands_data.reshape((rows * cols, n_bands))
    to = pd.DataFrame(flat_pixels)

    # to = None
    to1 = None

    to1 = pd.DataFrame({'Blue': to[0], 'Green': to[1], 'Red': to[2], 'NIR': to[3]})

    NDVI = (to1['NIR'] - to1['Red']) / (to1['NIR'] + to1['Red'])
    DVI = (to1['NIR'] - to1['Red'])
    to1['NDVI'] = NDVI
    to1['DVI'] = DVI

    file_exist_flag = os.path.exists(all_data_csv_file)
    if not file_exist_flag:
        to1.to_csv(all_data_csv_file, index=None)
    # 空值填充成0
    to1['NDVI'].fillna(int(0), inplace=True)

    #################################process_shp######################################
    labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
    # np.nonzero  参考 https://blog.csdn.net/u010315668/article/details/80204973
    is_train = np.nonzero(labeled_pixels)

    a_labels = labeled_pixels[is_train]
    print(bands_data)
    print(is_train)
    a_samples = bands_data[is_train]
    print(len(a_labels))
    for i in range(len(a_labels)):
        if a_labels[i] == 5:
            a_labels[i] = 3
    print(a_labels)
    print(a_samples)
    # a_samples = np.vstack(a_samples, a_labels)

    # a_samples = merge_w_np(a_samples, a_labels)
    a_labels = np.zeros(a_labels.shape[0])
    a_samples = np.column_stack((a_samples, a_labels))
    print(a_samples)
    file_exist_flag = os.path.exists(shape_files_csv_file)
    if not file_exist_flag:
        pd.DataFrame(a_samples).to_csv(shape_files_csv_file, index=None)

    print('####################################################################')
    all_data = pd.read_csv(all_data_csv_file)
    tol = all_data.iloc[:, :]
    tol[np.isinf(tol)] = 10
    tol[np.isnan(tol)] = 0
    print(tol)

    a_samples = np.array(a_samples)
    a_samples[np.isnan(a_samples)] = 0

    classifier = RandomForestClassifier(n_jobs=-1)
    classifier.fit(a_samples, a_labels)
    a = CVS(classifier, a_samples, a_labels, cv=10).mean()
    print(a)


def test_tif():
    """
    得到棉花、植被非棉花、非植被数据
    """
    raster_data_path_1 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_blue_b2.tif"
    raster_data_path_2 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_green_b3.tif"
    raster_data_path_3 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_red_b4.tif"
    raster_data_path_4 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_nir_b5.tif"

    all_data_csv_file = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_all_data_.csv"

    # raster_data_path_1 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/roi_hutubi_Red_20190914.tif"
    # raster_data_path_2 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/roi_hutubi_Green_20190914.tif"
    # raster_data_path_3 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/roi_hutubi_Blue_20190914.tif"
    # raster_data_path_4 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/roi_hutubi_NIR_20190914.tif"

    # shapefiles = ['/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/google_earth/cotton/xinjiang_shawan/xinjiang_shawan_cotton.shp',
    #               '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/google_earth/zhibei_uncotton/xinjiang_shawan_zhibei_uncotton/xinjiang_shawan_zhibei_uncotton.shp',
    #               '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/google_earth/unzhibei/xinjiang_shawan_unzhibei/xinjiang_shawan_unzhibei.shp',]

    # shapefiles = ['/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/google_earth/cotton/xinjiang_shawan/xinjiang_shawan_cotton.shp']
    # shape_files_csv_file = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_cotton.csv'

    # shapefiles = ['/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/google_earth/zhibei_uncotton/xinjiang_shawan_zhibei_uncotton/xinjiang_shawan_zhibei_uncotton.shp']
    # shape_files_csv_file = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_zhibei_uncotton.csv'

    # shapefiles = [
    #     '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/google_earth/unzhibei/xinjiang_shawan_unzhibei/xinjiang_shawan_unzhibei.shp']
    # shape_files_csv_file = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_unzhibei.csv'

    shapefiles = [
        '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/google_earth/unzhibei/xinjiang_shawan_unzhibei/xinjiang_shawan_unzhibei.shp']
    shape_files_csv_file = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_unzhibei.csv'

    output_fname = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/classification.tiff"

    tif(raster_data_path_1, raster_data_path_2, raster_data_path_3, raster_data_path_4, all_data_csv_file, shapefiles,
        shape_files_csv_file, output_fname)


def train():
    all_data = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_all_data_.csv'
    cotton = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_cotton.csv.csv'
    zhibei_uncotton = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_zhibei_uncotton.csv'
    unzhibei = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_unzhibei.csv'
    all_data = pd.read_csv(all_data)
    tol = all_data.iloc[:, :]
    tol[np.isinf(tol)] = 10
    tol[np.isnan(tol)] = 0
    print(tol)


def merge_multi_csv():
    """
    棉花、植被非棉花、非植被数据合并
    """
    cotton = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_cotton.csv'
    zhibei_uncotton = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_zhibei_uncotton.csv'
    unzhibei = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_unzhibei.csv'
    all_csv = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_all.csv'

    cotton = pd.read_csv(cotton, header=0)
    zhibei_uncotton = pd.read_csv(zhibei_uncotton, header=0)
    unzhibei = pd.read_csv(unzhibei, header=0)

    print(cotton)
    print(zhibei_uncotton)
    print(unzhibei)

    result = np.row_stack((cotton, zhibei_uncotton, unzhibei))
    print(result)


    file_exist_flag = os.path.exists(all_csv)
    if not file_exist_flag:
        pd.DataFrame(result).to_csv(all_csv, index=None, header=['Blue', 'Green', 'Red', 'NIR', 'label'])


def get_test_csv():
    """
    得到test数据
    """
    all_csv = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_file_all.csv'
    test_csv = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my/xinjiang_shawan_csv_test.csv'
    tool = pd.read_csv(all_csv)
    NDVI = (tool['NIR'] - tool['Red']) / (tool['NIR'] + tool['Red'])
    DVI = (tool['NIR'] - tool['Red'])
    tool['NDVI'] = NDVI
    tool['DVI'] = DVI
    file_exist_flag = os.path.exists(test_csv)
    if not file_exist_flag:
        tool.to_csv(test_csv, index=None)


if __name__ == '__main__':
    test_tif()
    # get_test_csv()
    # merge_multi_csv()
