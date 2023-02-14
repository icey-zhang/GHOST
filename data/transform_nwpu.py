import os
import pandas as pd
import cv2
import random
sets=['train','val'] #val test
root_path = '/home/data4/zjq/NWPU VHR-10 dataset'
ann_path = root_path+ '/ground truth'
output_path = root_path+ '/labels'
im_path = root_path+ '/positive image set'
ann_list = os.listdir(ann_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
train_rate = 0.8


def train_test_split():
    image_list = os.listdir(im_path) # 获取图片的原始路径
    image_number = len(image_list)
    

    train_number = int(image_number * train_rate)
    train_sample = random.sample(image_list, train_number) # 从image_list中随机获取0.8比例的图像.
    test_sample = list(set(image_list) - set(train_sample))
    sample = [train_sample, test_sample]
    return sample

def convert_annotation():
    for index, ann_filename in enumerate(ann_list):
        ann_filepath = os.path.join(ann_path, ann_filename)
        ann_df = pd.read_csv(ann_filepath, header=None)
        annstr = ''
        for i, ann in ann_df.iterrows():
            img_name = ann_filename[0:-3]+'jpg'
            img = cv2.imread(os.path.join(im_path, img_name))
            width = img.shape[1]
            height = img.shape[0]
            x1 = int(ann[0][1:])
            y1 = int(ann[1][0:-1])
            x2 = int(ann[2][1:])
            y2 = int(ann[3][0:-1])
            label = int(ann[4]) - 1
            x_center = (x1+x2)/2/width
            y_center = (y1+y2)/2/height
            w = (x2-x1)/width
            h = (y2-y1)/height
            annstr += f'{label} {x_center} {y_center} {w} {h}\n'
        with open(os.path.join(output_path, ann_filename),'w') as f:
            f.write(annstr)
        print(f'{index} th file done!')

# convert_annotation()
sample = train_test_split()
for index,image_set in enumerate(sets):
    list_file = open(root_path+'/%s.txt'%(image_set), 'a+')  # 新建一个 txt文件，用于存放 图片的绝对路径：/media/common/yzn_file/DataSetsH/VOC/VOCdevkit/VOC2007/JPEGImages/000001.jpg
    for image_id in sample[index]:
        list_file.write('%s/%s\n'%(im_path,image_id))  # 向 txt 文件中写入 一张图片的绝对路径
    list_file.close()
        
