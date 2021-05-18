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


# In[2]:


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    print(vector_data_path)
    print(r'C:\Users\77957\Documents\ArcGIS\%s'%(vector_data_path))
    data_source = gdal.OpenEx(r'C:\Users\77957\Documents\ArcGIS\%s'%(vector_data_path), gdal.OF_VECTOR)
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
        label = i+1
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

def tif(raster_data_path_1, raster_data_path_2, raster_data_path_3, raster_data_path_4, output_fname):
    bands_data = []
    '''
    raster_data_path_1 = "C:/Users/77957/Desktop/验证样本/roi_hutubi_Blue_20190914.tif"
    raster_data_path_2 = "C:/Users/77957/Desktop/验证样本/roi_hutubi_Green_20190914.tif"
    raster_data_path_3 = "C:/Users/77957/Desktop/验证样本/roi_hutubi_Red_20190914.tif"
    raster_data_path_4 = "C:/Users/77957/Desktop/验证样本/roi_hutubi_NIR_20190914.tif"
    output_fname = "D:/jkl/压缩包/data/classification.tiff"
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

    raster_dataset = gdal.Open(raster_data_path_1, gdal.GA_ReadOnly)
    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    bands_data = []
    print("波段数")
    print(raster_dataset.RasterCount)

    for b in range(1, raster_dataset.RasterCount+1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())

    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape

    bands_data.shape

    flat_pixels = bands_data.reshape((rows*cols, n_bands))
    to = pd.DataFrame(flat_pixels)

    to = None
    to1 = None

    to1 = pd.DataFrame({'Blue': to[0], 'Green': to[1], 'Red': to[2], 'NIR': to[3]})


    NDVI = (to1['NIR'] - to1['Red'])/(to1['NIR'] + to1['Red'])
    DVI = (to1['NIR'] - to1['Red'])
    to1['NDVI'] = NDVI
    to1['DVI'] = DVI

    to1.to_csv(r'C:\Users\77957\Desktop\aaa\newtestht.csv',index=None)
    to1['NDVI'].fillna(int(0), inplace=True)



#shapefiles = ['cotton/cotton.shp', 'uncotton/uncotton.shp', '非棉花\Export_Output.shp', '非棉花v2\Export_Output.shp']
shapefiles = ['uncotton_new\Export_Output.shp']
labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)

is_train = np.nonzero(labeled_pixels)

a_labels = labeled_pixels[is_train]

a_samples = bands_data[is_train]


# In[43]:


len(a_labels)


# In[ ]:


for i in range(len(a_labels)):
    if a_labels[i] == 5:
        a_labels[i] = 3


# In[ ]:


a_labels


# In[44]:


a_samples


# In[45]:


pd.DataFrame(a_samples).to_csv(r'C:\Users\77957\Desktop\新样本\uncotton.csv',index=None)


# In[ ]:


tool = pd.read_csv(r'C:/Users/77957/Desktop/aaa/allband_shuffle.csv')


# In[ ]:


tool


# In[ ]:


a_samples = tool.iloc[:,0:1]


# In[ ]:


a_labels = tool.iloc[:,-1]


# In[ ]:


a_samples


# In[ ]:


to1


# In[ ]:


tol = to1.replace(-np.inf, -1)


# In[ ]:


from sklearn.preprocessing import Imputer
tol = Imputer().fit_transform(tol)


# In[ ]:


to11= pd.read_csv(r'C:\Users\77957\Desktop\aaa\test.csv')


# In[ ]:


to11


# In[ ]:


to1 = to11.iloc[:,0:1]


# In[ ]:


to1 = np.array(to1)


# In[ ]:


np.isnan(to1)


# In[ ]:


to1[np.isnan(to1)] = 0


# In[ ]:


to1 = to1.astype(float)


# In[ ]:


to1


# In[ ]:


to1 = np.array(to1,dtype=np.float64)


# In[ ]:


to1[np.isinf(to1)] = 999


# In[ ]:


classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(a_samples, a_labels)


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS


# In[ ]:


CVS(classifier,a_samples,a_labels,cv = 10).mean()


# In[ ]:





# In[ ]:


n_samples = rows*cols
flat_pixels = bands_data.reshape((n_samples, n_bands))
result = classifier.predict(flat_pixels)
classification = result.reshape((rows, cols))


# In[ ]:


from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams
f = plt.figure()
rcParams ['figure.figsize'] = 100, 300

r = bands_data[:,:,2]
g = bands_data[:,:,1]
b = bands_data[:,:,0]
rgb = np.dstack([r,g,b])
f.add_subplot(1, 2, 1)
plt.imshow(rgb/255)
f.add_subplot(1, 2, 2)
plt.imshow(classification)


# In[ ]:


aa = classification.reshape(-1)
Sum = 0
for i in range(len(aa)):
    if aa[i] == 1:
        Sum += 1


# In[ ]:


Sum


# In[ ]:


classification


# In[ ]:


tool = pd.read_csv(r'C:/Users/77957/Desktop/aaa/allband_new.csv')


# In[ ]:


tool


# In[ ]:


NDVI=(NIR-Red)/(NIR+R)
DVI=NIR-Red


# In[ ]:


data = pd.read_csv(r'C:/Users/77957/Desktop/样本/uncotton_4.csv')


# In[ ]:


data


# In[ ]:


data


# In[ ]:


data.drop_duplicates(keep='first',inplace=True)


# In[ ]:


data


# In[ ]:


data.to_csv(r'C:/Users/77957/Desktop/样本/uncotton_4.csv',index =None)


# In[ ]:


data1 = pd.read_csv(r'C:/Users/77957/Desktop/样本/.csv')


# In[ ]:




