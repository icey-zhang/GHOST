import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=['train','val']
# mode = 'test' #val test
root_path = '/home/data4/zjq/DIOR'
image_path = ['JPEGImages-trainval','JPEGImages-test']
classes = ["airplane", 
            "airport",
            "baseballfield",
            "basketballcourt",
            "bridge",
            "chimney",
            "dam",
            "Expressway-Service-area",
            "Expressway-toll-station",
            "golfcourse",
            "golffield",
            "groundtrackfield",
            "harbor",
            "overpass",
            "ship",
            "stadium",
            "storagetank",
            "tenniscourt",
            "trainstation",
            "vehicle",
            "windmill"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):  # 转换这一张图片的坐标表示方式（格式）,即读取xml文件的内容，计算后存放在txt文件中。
    in_file = open(root_path+'/Annotations'+'/%s.xml'%(image_id))
    out_file = open(root_path+'/labels'+'/%s.txt'%(image_id), 'w')
    print(image_id)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult) == 1:
            # continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# wd = getcwd()

# for index,image_set in enumerate(sets):
#     if not os.path.exists(root_path + '/labels/'):
#         os.makedirs(root_path+'/labels/')  # 新建一个 label 文件夹，用于存放yolo格式的标签文件：000001.txt
#     image_ids = open(root_path +"/ImageSets/Main"+'/%s.txt'%(image_set)).read().strip().split()  # 读取txt文件中 存放的图片的 id：000001hy-nas/mydata/xml/val.txt
#     list_file = open(root_path+'/%s.txt'%(image_set), 'a+')  # 新建一个 txt文件，用于存放 图片的绝对路径：/media/common/yzn_file/DataSetsH/VOC/VOCdevkit/VOC2007/JPEGImages/000001.jpg
#     for image_id in image_ids:
#         list_file.write('%s/%s.jpg\n'%(root_path+'/'+image_path[index],image_id))  # 向 txt 文件中写入 一张图片的绝对路径
#         # list_file.close()
#         convert_annotation(image_id)  # 转换这一张图片的坐标表示方式（格式）
#     list_file.close()

for index,image_set in enumerate(sets):
    if not os.path.exists(root_path + '/labels/'):
        os.makedirs(root_path+'/labels/')  # 新建一个 label 文件夹，用于存放yolo格式的标签文件：000001.txt
    # image_ids = open(root_path +"/ImageSets/Main"+'/%s.txt'%(image_set)).read().strip().split()  # 读取txt文件中 存放的图片的 id：000001hy-nas/mydata/xml/val.txt
    image_ids = listdir(root_path+'/'+image_path[index])
    list_file = open(root_path+'/%s.txt'%(image_set), 'a+')  # 新建一个 txt文件，用于存放 图片的绝对路径：/media/common/yzn_file/DataSetsH/VOC/VOCdevkit/VOC2007/JPEGImages/000001.jpg
    for image_id in image_ids:
        list_file.write('%s/%s\n'%(root_path+'/'+image_path[index],image_id))  # 向 txt 文件中写入 一张图片的绝对路径
        # list_file.close()
        convert_annotation(image_id)  # 转换这一张图片的坐标表示方式（格式）
    list_file.close()
