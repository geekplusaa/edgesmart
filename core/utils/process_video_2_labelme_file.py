# -*- coding: UTF-8 -*-
import base64
import json
from PIL import Image
from io import BytesIO
import requests
import cv2
import os
import numpy as np

"""
自动化标注工具类(windows环境)
功能:接入视频流,调用口罩模型检测,把视频流中戴口罩的数据全部自动标注.
作用:机器标注代替人工标注
"""


def img_2_base64(image):
    """
    图片转base64
    :param image:
    :return:
    """
    with open(image, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        return base64_data.decode()


def get_detection_response(frame, api_url):
    """
    获取检测结果
    :param frame:frame
    :param api_url:api接口
    :return:
    """
    img = Image.fromarray(frame)
    output_buffer = BytesIO()  # 创建一个BytesIO
    img.save(output_buffer, format='JPEG')  # 写入output_buffer
    byte_data = output_buffer.getvalue()  # 在内存中读取
    image_base64 = base64.b64encode(byte_data).decode()  # 转为BASE64
    data = json.dumps(
        {"image": image_base64, "visual_result": "True", "username": "xsrt", "password": "dGVzdC1wd2QxMjM="})
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, data=data, headers=headers)
    result = response.text
    result = json.loads(result)
    return result


def get_file(file_name, labelme_date, frame, file_path, flag):
    """
    把frame和labelme_data写到本地
    :param file_name: 文件名without后缀
    :param labelme_date: json文件内容
    :param frame: 图片内容
    :param file_path: 存盘路径
    :param flag: 是否包含指定的类别[face_mask]
    :return:
    """
    if flag:
        try:
            # 写入xml文件
            with open(file_path + '/mask/' + file_name + '.json', "w", encoding='utf-8') as f:
                f.write(labelme_date)
                f.close()
            # 保存图片
            cv2.imwrite(file_path + '/mask/' + file_name + '.jpg', frame)
        except Exception as err:
            print(err)
    else:
        try:
            # 写入xml文件
            with open(file_path + '/face/' + file_name + '.json', "w", encoding='utf-8') as f:
                f.write(labelme_date)
                f.close()
            # 保存图片
            cv2.imwrite(file_path + '/face/' + file_name + '.jpg', frame)
        except Exception as err:
            print(err)


def get_json(json_str, file_name):
    """
    模型返回的json生成labelme对应的json文件
    :param json:
    :return:
    """
    # json_text = json.text
    # result = json.loads(json)
    result = json_str
    result_str = {}
    data = json.loads(json.dumps(result_str))
    data["version"] = "4.5.6"
    data["flags"] = {}

    class_flag = False
    shapes = None
    if "data" in result:
        shapes = []
        # shapes = json.loads(json.dumps(shapes))
        result_data = result["data"]
        if len(result_data) > 0:
            for data_i in result_data:
                shape = {}
                shape = json.loads(json.dumps(shape))
                # data_i 是字典
                if "type" in data_i:
                    data_i_type = data_i["type"]
                    print(data_i_type)
                    if data_i_type == 'face_mask':
                        class_flag = True
                    shape["label"] = data_i_type

                    # points
                    points = []
                    # points = json.loads(json.dumps(points))
                    xmax = data_i["xmax"]
                    xmin = data_i["xmin"]
                    ymax = data_i["ymax"]
                    ymin = data_i["ymin"]
                    a_point = [float(xmin), float(ymin)]
                    b_point = [float(xmin), float(ymax)]
                    c_point = [float(xmax), float(ymax)]
                    d_point = [float(xmax), float(ymin)]
                    points.append(a_point)
                    points.append(b_point)
                    points.append(c_point)
                    points.append(d_point)

                    # points
                    shape["points"] = points

                    # group_id
                    shape["group_id"] = "null"
                    shape["shape_type"] = "polygon"
                    shape["flags"] = {}

                    shapes.append(shape)
                else:
                    return class_flag, None
    else:
        return class_flag, None
    # shapes
    data["shapes"] = shapes
    # imagePath
    imagePath = file_name + '.jpg'
    data["imagePath"] = imagePath
    # 图片base64编码
    base64_img = None
    if "base64" in result:
        base64_img = result["base64"]
    # imageData
    data["imageData"] = base64_img
    data["imageHeight"] = 300
    data["imageWidth"] = 300

    data = json.dumps(data)
    return class_flag, data


class process_video_2_labelme_file(object):
    """
    读取摄像头的视频流，自动生成labelme格式的标注文件和图片
    """

    def __init__(self, rtsp, process_flag, api_url, fps, base_path, show_video_flag, fix_file_name):
        """
        初始化一些参数
        :param rtsp: 视频流源地址
        :param process_flag: 是否开启自动标注
        :param api_url: 模型接口地址
        :param fps: 每隔多少帧处理一次
        :param base_path: 存盘路径
        :param show_video_flag: 弹窗show开关
        :param fix_file_name: 文件名前缀
        """
        self.rtsp = rtsp
        self.process_flag = process_flag
        self.api_url = api_url
        self.fps = fps
        self.base_path = base_path
        self.show_video_flag = show_video_flag
        self.fix_file_name = fix_file_name

    def process_video(self):
        cap = cv2.VideoCapture(self.rtsp)
        print("publishing video...")
        i = 1
        a = self.fps
        while True:
            i = i + 1
            if i % a == 0:
                flag, frame = cap.read()
                if not flag:
                    break
                if self.process_flag:
                    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2RGBA)
                    json_str = get_detection_response(frame2, self.api_url)
                    # 生成labelme格式的json文件
                    file_name = self.fix_file_name + '' + str(i - 1)
                    flag, labelme_date = get_json(json_str, file_name)
                    # 生成文件
                    frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2RGBA)
                    if labelme_date is not None:
                        get_file(file_name, labelme_date, frame, self.base_path, flag)

                if self.show_video_flag:
                    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                    cv2.imshow("result", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

        cap.release()


def create_labelme_file():
    rtsp = 'D:/ai/data/mask/video/7.mp4'
    process_flag = True
    # api_url = 'http://localhost:8188/mask/json'
    # api_url = 'http://1.119.197.163:19899/mask/json'
    api_url = 'http://192.168.20.99:9999/mask/json'
    fps = 1
    base_path = os.getcwd() + '/data_result'
    show_video_flag = True
    fix_file_name = '20210305_mask_7_'
    fps = int(fps)
    if rtsp == '0':
        rtsp = 0
    print('rtsp=' + str(rtsp))
    print('api_url=' + api_url)
    print('fps=' + str(fps))
    print('process_flag=' + str(process_flag))
    process_class = process_video_2_labelme_file(rtsp, process_flag, api_url, fps, base_path, show_video_flag,
                                                 fix_file_name)
    process_class.process_video()


if __name__ == "__main__":
    create_labelme_file()
