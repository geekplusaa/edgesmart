#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import os
from sklearn import metrics
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as XGBR
# A list of "random" colors (for a nicer output)
COLORS = ["#000000", "#DDDD00", "#1CE6FF", "#EE34FF", "#EE4A46", "#00EA41"]
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS


# In[5]:


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
    
    
    


# In[27]:


raster_data_path_1 = "C:/Users/77957/Desktop/tif9月/roi_shawan_Blue_20190921.tif"
raster_data_path_2 = "C:/Users/77957/Desktop/tif9月/roi_shawan_Green_20190921.tif"
raster_data_path_3 = "C:/Users/77957/Desktop/tif9月/roi_shawan_Red_20190921.tif"
raster_data_path_4 = "C:/Users/77957/Desktop/tif9月/roi_shawan_NIR_20190921.tif"

data_list = [raster_data_path_1,raster_data_path_2,raster_data_path_3,raster_data_path_4]

output_fname = "D:/jkl/压缩包/data/classification.tiff"


# In[3]:


raster_data_path_1 = "C:/Users/77957/Desktop/验证样本/roi_hutubi_Blue_20190914.tif"
raster_data_path_2 = "C:/Users/77957/Desktop/验证样本/roi_hutubi_Green_20190914.tif"
raster_data_path_3 = "C:/Users/77957/Desktop/验证样本/roi_hutubi_Red_20190914.tif"
raster_data_path_4 = "C:/Users/77957/Desktop/验证样本/roi_hutubi_NIR_20190914.tif"

data_list = [raster_data_path_1,raster_data_path_2,raster_data_path_3,raster_data_path_4]

output_fname = "D:/jkl/压缩包/data/classification.tiff"


# In[28]:


bands_data = []
for i in range(len(data_list)):
    
    raster_dataset = gdal.Open(data_list[i], gdal.GA_ReadOnly)
    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()

    band = raster_dataset.GetRasterBand(1)
    bands_data.append(band.ReadAsArray())

    
bands_data = np.dstack(bands_data)
rows, cols, n_bands = bands_data.shape


# In[29]:


tool = pd.read_csv(r'C:/Users/77957/Desktop/新样本/样本合并.csv')


# In[30]:


tool


# In[ ]:


NDVI = (tool['NIR'] - tool['Red'])/(tool['NIR'] + tool['Red'])
DVI = (tool['NIR'] - tool['Red'])
tool['NDVI'] = NDVI
tool['DVI'] = DVI
tool.to_csv(r'C:\Users\77957\Desktop\新样本\test.csv',index=None)


# In[31]:


tool1 = pd.read_csv(r'C:/Users/77957/Desktop/新样本/test_shuffle_shot.csv')


# In[32]:


tool1


# In[33]:


a_samples = tool1.iloc[:,:-1]
a_labels = tool1.iloc[:,-1]


# In[34]:


a_samples


# In[35]:


to11= pd.read_csv(r'C:/Users/77957/Desktop/aaa/newtest.csv')
tol = to11.iloc[:,:]


# In[36]:


tol[np.isinf(tol)] = 10


# In[37]:


tol[np.isnan(tol)] = 0


# In[38]:


tol


# In[39]:


a_samples = np.array(a_samples)
a_samples[np.isnan(a_samples)] = 0


# In[40]:


classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(a_samples, a_labels)


# In[41]:


CVS(classifier,a_samples,a_labels,cv = 10).mean()


# In[ ]:





# In[42]:


# n_samples = rows*cols

# flat_pixels = bands_data.reshape((n_samples, n_bands))
result = classifier.predict(tol)
classification = result.reshape((rows, cols))


# In[43]:


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
plt.imshow(classification,cmap='gray')


# In[44]:


aa = classification.reshape(-1)
Sum = 0
for i in range(len(aa)):
    if aa[i] == 1:
        Sum += 1


# In[45]:


Sum


# In[ ]:




