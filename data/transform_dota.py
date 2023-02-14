'''
DOTA-v1.0:
|____val:
......|_images:
.........|_P0003.png
.........|_...
.........|_P2802.png
......|_labelTxt-v1.0
.........|_labelTxt
.........|_Train_Task2_gt #水平框
......|_labelTxt-v1.5:
.........|_DOTA-v1.5_val:
............|_P0003.txt
............|_...
............|_P2802.txt
.........|_DOTA-v1.5_val_hbb:
............|_P0003.txt
............|_...
............|_P2802.txt
|____train
|____test
'''
from DOTA_devkit_YOLO.ImgSplit import splitbase
from DOTA_devkit_YOLO.YOLO_Transform import dota2Darknet
from DOTA_devkit_YOLO.SplitOnlyImage import splitbase as splitbase_image
import os

root_path = '/home/data4/zjq/DOTA-v1.0'
save_path =  '/home/data4/zjq/DOTA-v1.0_split'

sets=['train','val','test']
# sets=['val']
split = splitbase(
                    basepath=root_path +'/train',
                      labelpath = 'labelTxt-v1.0/Train_Task2_gt/trainset_reclabelTxt',
                    outpath=save_path + '/train',
                    gap=200,
                    subsize=1024,
                    choosebestpoint=True,
           )
split.splitdata(rate=1)
split = splitbase(
                    basepath=root_path +'/val',
                    labelpath = 'labelTxt-v1.0/Val_Task2_gt/valset_reclabelTxt',
                    outpath=save_path + '/val',
                    gap=200,
                    subsize=1024,
                    choosebestpoint=True,
           )
split.splitdata(rate=1)
split = splitbase_image(
                    srcpath=root_path +'/test/images',
                    dstpath=save_path + '/test/images',
                    gap=200,
                    subsize=1024,
           )
split.splitdata(rate=1)
extractclassname = ['small-vehicle', 'large-vehicle', 'plane', 'storage-tank', 'ship',
 'harbor', 'ground-track-field','soccer-ball-field', 'tennis-court',
 'swimming-pool', 'baseball-diamond', 'roundabout', 'basketball-court', 
'bridge', 'helicopter']

for index,image_set in enumerate(sets):
    i = 0
    imgpath = '%s'%(save_path+'/'+image_set) +'/images'
    txtpath = '%s'%(save_path+'/'+image_set) +'/labelTxt'
    dstpath = '%s'%(save_path+'/'+image_set) +'/labels'
    dstimpath = '%s'%(save_path+'/'+image_set) +'/images'
    dota2Darknet(imgpath, txtpath, dstpath, extractclassname)
    image_list = os.listdir(dstimpath) # 获取图片的原始路径
    list_file = open(save_path+'/%s.txt'%(image_set), 'w')  # 新建一个 txt文件，用于存放 图片的绝对路径：/media/common/yzn_file/DataSetsH/VOC/VOCdevkit/VOC2007/JPEGImages/000001.jpg
    for image_id in image_list:
        i=i+1
        list_file.write('%s/%s\n'%(dstimpath,image_id))  # 向 txt 文件中写入 一张图片的绝对路径
    print('the length of images is: ',i)
    list_file.close()

# image_list = os.listdir(root_path+'/val/images') # 获取图片的原始路径
# list_file = open(root_path+'/%s.txt'%('val'), 'a')  # 新建一个 txt文件，用于存放 图片的绝对路径：/media/common/yzn_file/DataSetsH/VOC/VOCdevkit/VOC2007/JPEGImages/000001.jpg
# for image_id in image_list:
#     list_file.write('%s\n'%(image_id.replace('.png','')))  # 向 txt 文件中写入 一张图片的绝对路径
# list_file.close()

# image_list = os.listdir(save_path+'/val/images') # 获取图片的原始路径
# list_file = open(save_path+'/%s.txt'%('val1'), 'a')  # 新建一个 txt文件，用于存放 图片的绝对路径：/media/common/yzn_file/DataSetsH/VOC/VOCdevkit/VOC2007/JPEGImages/000001.jpg
# for image_id in image_list:
#     list_file.write('%s\n'%(image_id.replace('.png','')))  # 向 txt 文件中写入 一张图片的绝对路径
# list_file.close()
