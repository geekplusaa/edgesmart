# -*- coding:utf-8 -*-

import arcpy
import glob

import os
import numpy as np
# python读取遥感影像，写出遥感影像
from osgeo import gdal


def read_tif(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    band1 = dataset.GetRasterBand(1)
    print(band1)
    # 近红外波段
    im_data = dataset.GetRasterBand(1).ReadAsArray(0, 0, im_width, im_height)

    del dataset
    return im_data, im_width, im_height, im_geotrans, im_proj


def write_img(filename, im_proj, im_geotrans, im_data):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def multiimage_2_oneimage(band1_fn, band2_fn, band3_fn, band4_fn, out_name):
    """
    Landsat图像的波段1--->蓝色波段
    Landsat图像的波段2--->绿色波段
    Landsat图像的波段3--->红色波段
    Landsat图像的波段4--->近红外光波段
    Landsat图像的波段5--->中红外光波段
    Landsat图像的波段6--->热红外光波段
    Landsat图像的波段7--->红红外光波段
    """
    in_ds = gdal.Open(band1_fn)
    in_band = in_ds.GetRasterBand(1)

    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(out_name, in_band.XSize, in_band.YSize, 3, in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())

    # 读取第1波段数据
    in_data = in_band.ReadAsArray()
    out_band = out_ds.GetRasterBand(3)
    out_band.WriteArray(in_data)

    # 读取第2波段数据
    in_ds = gdal.Open(band2_fn)
    out_band = out_ds.GetRasterBand(2)
    out_band.WriteArray(in_ds.ReadAsArray())

    # 读取第3波段数据
    out_ds.GetRasterBand(1).WriteArray(
        gdal.Open(band3_fn).ReadAsArray())

    out_ds.FlushCache()
    for i in range(1, 4):
        out_ds.GetRasterBand(i).ComputeStatistics(False)

    out_ds.BuildOverviews('average', [2, 4, 8, 16, 32])
    del out_ds


def multiimage_2_oneimage1():
    filename = r"/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/20190617-20200827"
    # 输出文件夹
    output_layerstack = r"/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result"
    files = []
    datas = []
    # listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    for file in os.listdir(filename):
        if os.path.splitext(file)[1] == '.TIF':
            sourcefile = os.path.join(filename, file)
            files.append(sourcefile)

    for file, k in zip(files, range(len(files))):
        # 读入矢量和栅格文件
        (filepath, tempfilename) = os.path.split(file)
        (filename, extension) = os.path.splitext(tempfilename)
        # out_file = outpath + '/' + filename + '.tif'
        print(filename)
        data, height, width, geotrans, proj = read_tif(file)
        print(data)
        datas.append(data)

    datas = np.array(datas)
    print(datas.shape)
    write_img(output_layerstack + "/layerstack1.tif", proj, geotrans, datas, )


def crop_tif(tif_file_path, shp_file, output_path):
    tif_file_path = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my'
    shp_file = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/google_earth/cotton/xinjiang_shawan/xinjiang_shawan_cotton.shp"
    output_path = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/my"
    # 利用glob包，将inws下的所有tif文件读存放到rasters中
    rasters = glob.glob(os.path.join(tif_file_path, "*.tif"))
    mask = shp_file
    # 循环rasters中的所有影像，进行按掩模提取操作
    for ras in rasters:
        outname = os.path.join(output_path,
                               os.path.basename(ras).split(".")[0] + "_clp.tif")  # 指定输出文件的命名方式（以被裁剪文件名+_clip.tif命名）
        out_extract = arcpy.sa.ExtractByMask(ras, mask)  # 执行按掩模提取操作
        out_extract.save(outname)  # 保存数据


if __name__ == '__main__':
    raster_data_path_1 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/20190617-20200827/LC08_L2SP_144029_20190617_20200827_02_T1_SR_B2.TIF"
    raster_data_path_2 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/20190617-20200827/LC08_L2SP_144029_20190617_20200827_02_T1_SR_B3.TIF"
    raster_data_path_3 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/20190617-20200827/LC08_L2SP_144029_20190617_20200827_02_T1_SR_B4.TIF"
    raster_data_path_4 = "/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/20190617-20200827/LC08_L2SP_144029_20190617_20200827_02_T1_SR_B5.TIF"
    out_name = '/home/geekplusa/ai/datasets/xraybot/remote_Sensing/result/20190617-20200827/nat_color.tif'
    # multiimage_2_oneimage(raster_data_path_1, raster_data_path_2, raster_data_path_3, raster_data_path_4, out_name)
    multiimage_2_oneimage1()
