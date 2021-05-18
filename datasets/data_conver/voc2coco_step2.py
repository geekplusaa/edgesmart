# coding:utf-8

# pip install lxml

import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import cv2

cocopath = r"../cocodataset/"

START_BOUNDING_BOX_ID = 1


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def get(root, name):
    return root.findall(name)


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_list, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    categories_num = pre_define_categories_numbers.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    for index, line in enumerate(xml_list):
        # print("Processing %s"%(line))
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()

        filename = os.path.basename(xml_f)[:-4] + ".jpg"
        image_id = 20200904001 + index

        xmlname = xml_f.split(".xml")[0]
        jpgname = xmlname + ".jpg"
        img = cv2.imread(jpgname)
        print(xmlname)
        print(jpgname)
        width = img.shape[1]
        height = img.shape[0]

        image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}

        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        valide = True
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category in all_categories:
                all_categories[category] += 1
            else:
                all_categories[category] = 1
            if category not in categories:
                if only_care_pre_define_categories:
                    continue
                new_id = len(categories) + 1
                print(
                    "[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(
                        category, pre_define_categories, new_id))
                categories[category] = new_id
            categories_num[category] += 1
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            if xmax <= xmin:
                valide = False
                continue
            if ymax <= ymin:
                valide = False
                continue
            # assert(xmax > xmin), "xmax <= xmin, {}".format(line)
            # assert(ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        if valide:
            json_dict['images'].append(image)

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, indent=4, cls=MyEncoder)
    # json.dumps(json_dict,open(json_file, 'w'))
    # json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))
    print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories),
                                                                                  all_categories.keys(),
                                                                                  len(pre_define_categories),
                                                                                  pre_define_categories.keys()))
    print("category: id --> {}".format(categories))
    print(categories.keys())
    print(categories.values())
    print("all categories : {}".format(all_categories))


if __name__ == '__main__':
    classes = [
        'face',
        'face_mask']
    pre_define_categories = {}
    pre_define_categories_numbers = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1
        pre_define_categories_numbers[cls] = 0
    # pre_define_categories = {'a1': 1, 'a3': 2, 'a6': 3, 'a9': 4, "a10": 5}
    only_care_pre_define_categories = True
    # only_care_pre_define_categories = False

    # train_ratio = 0.9
    save_json_train = cocopath + 'annotations/instances_train.json'
    save_json_val = cocopath + 'annotations/instances_val.json'
    # save_json_test = cocopath+'annotations/test.json'
    # xml_dir = "./test"

    # xml_list = glob.glob(r"resize/*.xml")
    # xml_list = np.sort(xml_list)
    # np.random.seed(100)
    # n#p.random.shuffle(xml_list)

    # train_num = int(len(xml_list)*train_ratio)
    # xml_list_train = xml_list[:train_num]
    xml_list_train = glob.glob(r"../train_val/train/*.xml")
    # xml_list_val = xml_list[train_num:]
    xml_list_val = glob.glob(r"../train_val/val/*.xml")
    # xml_list_test = glob.glob(r"dataset/test/*.xml")
    if os.path.exists(cocopath + "annotations"):
        shutil.rmtree(cocopath + "annotations")
    os.makedirs(cocopath + "annotations")
    convert(xml_list_train, save_json_train)
    convert(xml_list_val, save_json_val)
    # convert(xml_list_test, save_json_test)

    if os.path.exists(cocopath + "train"):
        shutil.rmtree(cocopath + "train")
    os.makedirs(cocopath + "train")
    if os.path.exists(cocopath + "val"):
        shutil.rmtree(cocopath + "val")
    os.makedirs(cocopath + "val")

    if os.path.exists(cocopath + "test"):
        shutil.rmtree(cocopath + "test")
    os.makedirs(cocopath + "test")

    f1 = open(cocopath + "train.txt", "w")
    for xml in xml_list_train:
        img = xml[:-4] + ".jpg"
        f1.write(os.path.basename(xml)[:-4] + "\n")
        shutil.copyfile(img, cocopath + "train/" + os.path.basename(img))

    f2 = open(cocopath + "val.txt", "w")
    for xml in xml_list_val:
        img = xml[:-4] + ".jpg"
        f2.write(os.path.basename(xml)[:-4] + "\n")
        shutil.copyfile(img, cocopath + "val/" + os.path.basename(img))

    #    f3 = open(cocopath+"test.txt", "w")
    #    for xml in xml_list_test:
    #        img = xml[:-4] + ".jpg"
    #        f3.write(os.path.basename(xml)[:-4] + "\n")
    #        shutil.copyfile(img, cocopath + "test/" + os.path.basename(img))
    f1.close()
    f2.close()
    # f3.close()
    print("-------------------------------")
    print("train number:", len(xml_list_train))
    print("val number:", len(xml_list_val))
# print("test number:", len(xml_list_test))