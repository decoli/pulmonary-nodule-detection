import argparse
import copy
import csv
import glob
import os
import random
import sys
import xml.etree.ElementTree as ET
from glob import glob
from random import shuffle
from xml.etree.ElementTree import Element, ElementTree

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk  # 医疗图片处理包
from PIL import Image
from skimage import measure, morphology
from skimage.transform import resize
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(
    description='use data for debug'
)
parser.add_argument('--debug', action='store_true', help='use data for debug')
parser.add_argument('--draw_nodule', action='store_true', help='draw the location of nodule')
parser.add_argument('--nodule_size', default=32, type=int, help='set the GT of size of nodule') # default 0 for dynamic size
parser.add_argument('--bigger_size', default=1, type=float, help='set the GT size bigger')
parser.add_argument('--times_movement', default=1, type=int, help='set the times nodule movement for a CT image')

parser.add_argument(
    '--mode',choices=[

        # make the image to feed the model
        'get_masked_image',
        'negative_get_masked_image',
        'negative_masked_image_rename',

        'get_voc_anno',
        'negative_get_voc_anno',
        'check_multi_nodule',

        # get text.txt/trainval.txt in ImageSets/Main
        'get_main_txt',
        'negative_get_main_txt',

        'augmentation_movement'
        ],help='set the run mode'
    )

args = parser.parse_args()

# set path

if os.name == 'posix':
    working_path = '/Volumes/shirui_WD_2/lung_image/all_LUNA16/LUNA16'
else:
    working_path = 'E:\lung_image\\all_LUNA16\\LUNA16'

cand_path = 'data/LUNA16/candidates.csv'
anno_path = 'data/LUNA16/annotations.csv'
negative_anno_path = 'data/LUNA16/negative/negative_anno.csv'

'''convert world coordinate to real coordinate'''
def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(itkimage.GetOrigin()))  # CT原点坐标
    numpySpacing = np.array(list(itkimage.GetSpacing()))  # CT像素间隔
    return numpyImage, numpyOrigin, numpySpacing

def get_masked_image():
    pd_annotation = pd.read_csv(anno_path)
    count_image = 0
    for each_annotation in pd_annotation.iterrows():
        seriesuid = each_annotation[1].seriesuid
        coord_x = each_annotation[1].coordX
        coord_y = each_annotation[1].coordY
        coord_z = each_annotation[1].coordZ
        diameter_mm = each_annotation[1].diameter_mm

        mhd_name = '{}.mhd'.format(seriesuid)
        mhd_path = glob(os.path.join(working_path, '*', mhd_name), recursive=True)[0]

        numpyImage, numpyOrigin, numpySpacing = load_itk_image(mhd_path) # numpyImage.shape) 维度为(slice,w,h)

        # 将世界坐标下肺结节标注转换为真实坐标系下的坐标标注
        worldCoord = np.asarray([float(coord_x),float(coord_y),float(coord_z)])
        voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    
        slice = int(voxelCoord[2] + 0.5)
        img = np.squeeze(numpyImage[slice, ...])  # if the img is 3d, the slice is integer

        # 由于肺部与周围组织颜色对比明显，考虑通过聚类的方法找到可区分肺区域和非肺区域的阈值，实现二值化。
        mean = np.mean(img)
        std = np.std(img)
        img = img-mean
        img = img/std

        # K-mean
        #提取肺部大致均值
        middle = img[100:400,100:400]  
        mean = np.mean(middle)  

        # 将图片最大值和最小值替换为肺部大致均值
        max = np.max(img)
        min = np.min(img)
        img[img==max]=mean
        img[img==min]=mean

        # image_array = img
        # f, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,8))
        # ax1.imshow(image_array,cmap='gray')
        # ax2.hist(img.flatten(),bins=200)
        # plt.show()

        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)  
        thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

        # 聚类完成后，清晰可见偏黑色区域为一类，偏灰色区域为另一类。
        # image_array = thresh_img
        # plt.imshow(image_array,cmap='gray') 
        # plt.show()

        eroded = morphology.erosion(thresh_img,np.ones([4,4]))  
        dilation = morphology.dilation(eroded,np.ones([10,10]))  
        labels = measure.label(dilation)   
        # fig,ax = plt.subplots(2,2,figsize=[8,8])
        # ax[0,0].imshow(thresh_img,cmap='gray')  
        # ax[0,1].imshow(eroded,cmap='gray') 
        # ax[1,0].imshow(dilation,cmap='gray')  
        # ax[1,1].imshow(labels)  # 标注mask区域切片图
        # plt.show()

        label_vals = np.unique(labels)
        regions = measure.regionprops(labels) # 获取连通区域

        # 设置经验值，获取肺部标签
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
    
        # 根据肺部标签获取肺部mask，并再次进行’膨胀‘操作，以填满并扩张肺部区域
        mask = np.ndarray([512,512],dtype=np.int8)
        mask[:] = 0
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

        # flag draw nodule
        if args.draw_nodule:
            point_left = int(voxelCoord[0] + 0.5) - 16
            point_up = int(voxelCoord[1] + 0.5) - 16
            point_left_up = (int(point_left), int(point_up))
            point_right = int(voxelCoord[0] + 0.5) + 16
            point_down = int(voxelCoord[1] + 0.5) + 16
            point_right_down = (int(point_right), int(point_down))
            cv2.rectangle(img, point_left_up, point_right_down, (0, 0, 255), 2)

        # save the image
        width = img.shape[0]
        height = img.shape[1]
        dpi = 100
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        axes = fig.add_axes([0, 0, 1, 1])
        axes.set_axis_off()
        axes.imshow(img*mask, cmap='gray')
        path_img = os.path.join('data/LUNA16/masked/{:06d}.png'.format(count_image))
        fig.savefig(path_img)
        plt.close()
        print(count_image)

        count_image += 1

def negative_get_masked_image():

    # save candidates data
    candidates_list = []
    candidates_list.append('seriesuid')
    candidates_list.append('coordX')
    candidates_list.append('coordY')
    candidates_list.append('coordZ')

    with open('data/LUNA16/negative/masked/negative_anno.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(candidates_list)

    pd_annotation = pd.read_csv(cand_path)
    pd_annotation = pd_annotation[pd_annotation['class'] == 0]
    pd_annotation = pd_annotation.sample(frac=1).reset_index(drop=True)

    count_image = 0
    for each_annotation in pd_annotation.iterrows():
        seriesuid = each_annotation[1].seriesuid
        coord_x = each_annotation[1].coordX
        coord_y = each_annotation[1].coordY
        coord_z = each_annotation[1].coordZ
        # diameter_mm = each_annotation[1].diameter_mm

        mhd_name = '{}.mhd'.format(seriesuid)
        mhd_path = glob(os.path.join(working_path, '*', mhd_name), recursive=True)[0]

        numpyImage, numpyOrigin, numpySpacing = load_itk_image(mhd_path) # numpyImage.shape) 维度为(slice,w,h)

        # 将世界坐标下肺结节标注转换为真实坐标系下的坐标标注
        worldCoord = np.asarray([float(coord_x),float(coord_y),float(coord_z)])
        voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    
        slice = int(voxelCoord[2] + 0.5)
        img = np.squeeze(numpyImage[slice, ...])  # if the img is 3d, the slice is integer

        # 由于肺部与周围组织颜色对比明显，考虑通过聚类的方法找到可区分肺区域和非肺区域的阈值，实现二值化。
        mean = np.mean(img)
        std = np.std(img)
        img = img-mean
        img = img/std

        # K-mean
        #提取肺部大致均值
        middle = img[100:400,100:400]  
        mean = np.mean(middle)  

        # 将图片最大值和最小值替换为肺部大致均值
        max = np.max(img)
        min = np.min(img)
        img[img==max]=mean
        img[img==min]=mean

        # image_array = img
        # f, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,8))
        # ax1.imshow(image_array,cmap='gray')
        # ax2.hist(img.flatten(),bins=200)
        # plt.show()

        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)  
        thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

        # 聚类完成后，清晰可见偏黑色区域为一类，偏灰色区域为另一类。
        # image_array = thresh_img
        # plt.imshow(image_array,cmap='gray') 
        # plt.show()

        eroded = morphology.erosion(thresh_img,np.ones([4,4]))  
        dilation = morphology.dilation(eroded,np.ones([10,10]))  
        labels = measure.label(dilation)   
        # fig,ax = plt.subplots(2,2,figsize=[8,8])
        # ax[0,0].imshow(thresh_img,cmap='gray')  
        # ax[0,1].imshow(eroded,cmap='gray') 
        # ax[1,0].imshow(dilation,cmap='gray')  
        # ax[1,1].imshow(labels)  # 标注mask区域切片图
        # plt.show()

        label_vals = np.unique(labels)
        regions = measure.regionprops(labels) # 获取连通区域

        # 设置经验值，获取肺部标签
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
    
        # 根据肺部标签获取肺部mask，并再次进行’膨胀‘操作，以填满并扩张肺部区域
        mask = np.ndarray([512,512],dtype=np.int8)
        mask[:] = 0
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

        # flag draw nodule
        if args.debug:
            point_left = int(voxelCoord[0] + 0.5) - 16
            point_up = int(voxelCoord[1] + 0.5) - 16
            point_left_up = (int(point_left), int(point_up))
            point_right = int(voxelCoord[0] + 0.5) + 16
            point_down = int(voxelCoord[1] + 0.5) + 16
            point_right_down = (int(point_right), int(point_down))
            cv2.rectangle(img, point_left_up, point_right_down, (0, 0, 255), 2)

        # save the image
        width = img.shape[0]
        height = img.shape[1]
        dpi = 100
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        axes = fig.add_axes([0, 0, 1, 1])
        axes.set_axis_off()
        axes.imshow(img*mask, cmap='gray')
        path_img = os.path.join('data/LUNA16/negative/masked/Annotations/{:06d}.png'.format(count_image + 100000))
        fig.savefig(path_img)
        plt.close()
        # print(count_image)

        # save candidates data
        candidates_list = []
        candidates_list.append(seriesuid)
        candidates_list.append(coord_x)
        candidates_list.append(coord_y)
        candidates_list.append(coord_z)

        with open('data/LUNA16/negative/masked/negative_anno.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(candidates_list)

        count_image += 1
        print('\rplease wait... {:.2%}'.format(count_image / 10000), end='', flush=True)
        if count_image == 10000:
            break
    print('')
    print(
        'output:\n'
        'data/LUNA16/negative/masked/negative_anno.csv\n'
        'data/LUNA16/negative/masked/JPEGImages/')

def negative_masked_image_rename():
    negative_masked_image_dir = 'data/LUNA16/negative/masked/JPEGImages'
    negative_masked_image_path_list = glob(os.path.join(negative_masked_image_dir, '*.png'))
    for each_image_path in negative_masked_image_path_list:
        image_name = int(os.path.basename(each_image_path).split('.')[0])
        image_rename = '{}.png'.format(int(image_name) + 100000)
        image_rename_path = os.path.join(negative_masked_image_dir, image_rename)
        os.rename(each_image_path, image_rename_path)

def get_voc_anno():
    def beatau(e,level=0):
        if len(e) > 0:
            e.text='\n'+'\t'*(level+1)
            for child in e:
                beatau(child,level+1)
            child.tail=child.tail[:-1]
        e.tail='\n' + '\t'*level
    
    def to_xml(name, list_x, list_y, list_w, list_h):
        root = Element('annotation')#根节点
        erow1 = Element('folder')#节点1
        erow1.text= "VOC"
        
        
        erow2 = Element('filename')#节点2
        erow2.text= str(name)
        
        erow3 = Element('size')#节点3
        erow31 = Element('width')
        erow31.text = "512"
        erow32 = Element('height')
        erow32.text = "512"
        erow33 = Element('depth')
        erow33.text = "3" 
        erow3.append(erow31)
        erow3.append(erow32)
        erow3.append(erow33)

        root.append(erow1)
        root.append(erow2)
        root.append(erow3)

        for x, y, w, h in zip(list_x, list_y, list_w, list_h):
            erow4 = Element('object')
            
            erow41 = Element('name')
            erow41.text = 'nodule'

            erow4_pos = Element('pose')
            erow4_pos.text = 'Unspecified'

            erow4_tru = Element('truncated')
            erow4_tru.text = '0'

            erow4_dif = Element('difficult')
            erow4_dif.text = '0'

            erow42 = Element('bndbox')

            erow4.append(erow41)
            erow4.append(erow4_pos)
            erow4.append(erow4_tru)
            erow4.append(erow4_dif)
            erow4.append(erow42)

            erow421 = Element('xmin')
            erow421.text = str(x - np.round(w/2).astype(int))

            erow422 = Element('ymin')
            erow422.text = str(y - np.round(h/2).astype(int))

            erow423 = Element('xmax')
            erow423.text = str(x + np.round(w/2).astype(int))

            erow424 = Element('ymax')
            erow424.text = str(y + np.round(h/2).astype(int))

            erow42.append(erow421)
            erow42.append(erow422)
            erow42.append(erow423)
            erow42.append(erow424)

            root.append(erow4)

        beatau(root)      

        return ElementTree(root)

    def write_xml(tree, out_path):  
        '''''将xml文件写出 
        tree: xml树 
        out_path: 写出路径'''  
        tree.write(out_path, encoding="utf-8",xml_declaration=True)

    pd_annotation = pd.read_csv(anno_path)

    seriesuid_temp = None
    nodule_uid_list = []
    nodule_dict_list = []

    count_image = 0
    print('output: data\\LUNA16\\masked\\Annotations\\')
    for each_annotation in pd_annotation.iterrows():
        seriesuid = each_annotation[1].seriesuid
        coord_x = each_annotation[1].coordX
        coord_y = each_annotation[1].coordY
        coord_z = each_annotation[1].coordZ
        diameter_mm = each_annotation[1].diameter_mm

        mhd_name = '{}.mhd'.format(seriesuid)

        try:
            mhd_path = glob(os.path.join(working_path, '*', mhd_name), recursive=True)[0]
        except IndexError:
            print(
                'no LUNA16 data.\n'
                'set ssd to usb.')
            sys.exit(0)
            
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(mhd_path) # numpyImage.shape) 维度为(slice,w,h)

        # 将世界坐标下肺结节标注转换为真实坐标系下的坐标标注
        worldCoord = np.asarray([float(coord_x),float(coord_y),float(coord_z)])
        voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    
        slice = int(voxelCoord[2] + 0.5)

        if args.nodule_size == 0: # dynamic
            nodule_dict = {
                'slice': slice,
                'x': int(voxelCoord[0] + 0.5),
                'y': int(voxelCoord[1] + 0.5),
                'w': int(diameter_mm / numpySpacing[0] * args.bigger_size + 0.5),
                'h': int(diameter_mm / numpySpacing[1] * args.bigger_size + 0.5),
                'count_image': count_image,
            }
        else:
            nodule_dict = {
                'slice': slice,
                'x': int(voxelCoord[0] + 0.5),
                'y': int(voxelCoord[1] + 0.5),
                'w': args.nodule_size * args.bigger_size,
                'h': args.nodule_size * args.bigger_size,
                'count_image': count_image,
            }

        if args.debug:
            masked_image_dir = 'data\LUNA16\masked\JPEGImages'
            name_image = '{:06d}.png'.format(nodule_dict['count_image'])
            masked_image_path = os.path.join(masked_image_dir, name_image)
            masked_image = cv2.imread(masked_image_path)
            cv2.imwrite('test/test.png', masked_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            point_left_up = (int(nodule_dict['x'] - nodule_dict['w'] / 2 + 0.5), int(nodule_dict['y'] - nodule_dict['h'] / 2 + 0.5))
            point_right_down = (int(nodule_dict['x'] + nodule_dict['w'] / 2 + 0.5), int(nodule_dict['y'] + nodule_dict['h'] / 2 + 0.5))
            cv2.rectangle(masked_image, point_left_up, point_right_down, (0, 0, 255), 1)
            cv2.imwrite('test/test.png', masked_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        # 针对第一个元素的处理
        if seriesuid_temp == None:
            nodule_dict_list.append(nodule_dict)
            seriesuid_temp = seriesuid

        else:
            if seriesuid_temp == seriesuid:
                nodule_dict_list.append(nodule_dict) # nodule_dict_list 增添元素 nodule_dict
            
            else:
                nodule_uid_list.append(nodule_dict_list) # nodule_uid_list 增添元素 nodule_dict_list

                nodule_dict_list = [] # 不能使用 nodule_dict_list.clear(), 会清空引用
                nodule_dict_list.append(nodule_dict) # 清空nodule_dict_list后，加入新uid的结节

                seriesuid_temp = seriesuid # 准备处理下一轮不同uid的结节数据

        print('\rplease wait... {:.2%}'.format((count_image + 1) / 1186), end='', flush=True)
        count_image +=1
    
    # 末尾数据的处理
    nodule_uid_list.append(nodule_dict_list)

    # loop nodule_uid_list
    for each_nodule_dict_list in nodule_uid_list:

        # get list of slice
        list_slice = []
        for each_nodule_dict in each_nodule_dict_list:
            if not each_nodule_dict['slice'] in list_slice:
                list_slice.append(each_nodule_dict['slice'])

        # make .xml for each nodule
        for each_slice in list_slice:

            list_nodule = []
            list_x = []
            list_y = []
            list_w = []
            list_h = []
            list_count_image = []

            for each_nodule_dict in each_nodule_dict_list:
                if each_nodule_dict['slice'] == each_slice:
                    list_nodule.append(each_nodule_dict)

            for each_nodule in list_nodule:
                list_x.append(each_nodule['x'])
                list_y.append(each_nodule['y'])
                list_h.append(each_nodule['h'])
                list_w.append(each_nodule['w'])
                list_count_image.append(each_nodule['count_image'])

            # make .xml
            name_image = '{}.png'.format(min(list_count_image))
            tree = to_xml(
                name=name_image,
                list_x=list_x,
                list_y=list_y,
                list_w=list_w,
                list_h=list_h
            )

            # save .xml
            write_xml(tree, "data\LUNA16\masked\Annotations\{:06d}.xml".format(min(list_count_image)))

def negative_get_voc_anno():
    def beatau(e,level=0):
        if len(e) > 0:
            e.text='\n'+'\t'*(level+1)
            for child in e:
                beatau(child,level+1)
            child.tail=child.tail[:-1]
        e.tail='\n' + '\t'*level
    
    def to_xml(name, list_x, list_y, list_w, list_h):
        root = Element('annotation')#根节点
        erow1 = Element('folder')#节点1
        erow1.text= "VOC"
        
        
        erow2 = Element('filename')#节点2
        erow2.text= str(name)
        
        erow3 = Element('size')#节点3
        erow31 = Element('width')
        erow31.text = "512"
        erow32 = Element('height')
        erow32.text = "512"
        erow33 = Element('depth')
        erow33.text = "3" 
        erow3.append(erow31)
        erow3.append(erow32)
        erow3.append(erow33)

        root.append(erow1)
        root.append(erow2)
        root.append(erow3)

        for x, y, w, h in zip(list_x, list_y, list_w, list_h):
            erow4 = Element('object')
            
            erow41 = Element('name')
            erow41.text = 'nonnodule'

            erow4_pos = Element('pose')
            erow4_pos.text = 'Unspecified'

            erow4_tru = Element('truncated')
            erow4_tru.text = '0'

            erow4_dif = Element('difficult')
            erow4_dif.text = '0'

            erow42 = Element('bndbox')

            erow4.append(erow41)
            erow4.append(erow4_pos)
            erow4.append(erow4_tru)
            erow4.append(erow4_dif)
            erow4.append(erow42)

            erow421 = Element('xmin')
            erow421.text = str(x - np.round(w/2).astype(int))

            erow422 = Element('ymin')
            erow422.text = str(y - np.round(h/2).astype(int))

            erow423 = Element('xmax')
            erow423.text = str(x + np.round(w/2).astype(int))

            erow424 = Element('ymax')
            erow424.text = str(y + np.round(h/2).astype(int))

            erow42.append(erow421)
            erow42.append(erow422)
            erow42.append(erow423)
            erow42.append(erow424)

            root.append(erow4)

        beatau(root)      

        return ElementTree(root)

    def write_xml(tree, out_path):  
        '''''将xml文件写出 
        tree: xml树 
        out_path: 写出路径'''  
        tree.write(out_path, encoding="utf-8",xml_declaration=True)

    pd_annotation = pd.read_csv(negative_anno_path)
    # pd_annotation = pd_annotation.sort_values(by='seriesuid')
    # pd_annotation.to_csv('data/LUNA16/negative/negative_anno_sorted_seriesuid.csv',index=0)

    seriesuid_temp = None
    nodule_uid_list = []
    nodule_dict_list = []

    count_image = 100000 # 100000开始为“非结节”
    print('output: data\\LUNA16\\negative\\masked\\Annotations\\')
    for each_annotation in pd_annotation.iterrows():
        seriesuid = each_annotation[1].seriesuid
        coord_x = each_annotation[1].coordX
        coord_y = each_annotation[1].coordY
        coord_z = each_annotation[1].coordZ

        mhd_name = '{}.mhd'.format(seriesuid)

        try:
            mhd_path = glob(os.path.join(working_path, '*', mhd_name), recursive=True)[0]
        except IndexError:
            print(
                'no LUNA16 data.\n'
                'set ssd to usb.')
            sys.exit(0)
            
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(mhd_path) # numpyImage.shape) 维度为(slice,w,h)

        # 将世界坐标下肺结节标注转换为真实坐标系下的坐标标注
        worldCoord = np.asarray([float(coord_x),float(coord_y),float(coord_z)])
        voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    
        slice = int(voxelCoord[2] + 0.5)

        nodule_dict = {
            'slice': slice,
            'x': int(voxelCoord[0] + 0.5),
            'y': int(voxelCoord[1] + 0.5),
            'w': args.nodule_size * args.bigger_size,
            'h': args.nodule_size * args.bigger_size,
            'count_image': count_image,
        }

        if args.debug:
            masked_image_dir = 'data/LUNA16/negative/masked/JPEGImages'
            name_image = '{:06d}.png'.format(nodule_dict['count_image'])
            masked_image_path = os.path.join(masked_image_dir, name_image)
            masked_image = cv2.imread(masked_image_path)
            cv2.imwrite('test/test.png', masked_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            point_left_up = (int(nodule_dict['x'] - nodule_dict['w'] / 2 + 0.5), int(nodule_dict['y'] - nodule_dict['h'] / 2 + 0.5))
            point_right_down = (int(nodule_dict['x'] + nodule_dict['w'] / 2 + 0.5), int(nodule_dict['y'] + nodule_dict['h'] / 2 + 0.5))
            cv2.rectangle(masked_image, point_left_up, point_right_down, (0, 0, 255), 1)
            cv2.imwrite('test/test.png', masked_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        # 针对第一个元素的处理
        if seriesuid_temp == None:
            nodule_dict_list.append(nodule_dict)
            seriesuid_temp = seriesuid

        else:
            if seriesuid_temp == seriesuid:
                nodule_dict_list.append(nodule_dict) # nodule_dict_list 增添元素 nodule_dict
            
            else:
                nodule_uid_list.append(nodule_dict_list) # nodule_uid_list 增添元素 nodule_dict_list

                nodule_dict_list = [] # 不能使用 nodule_dict_list.clear(), 会清空引用
                nodule_dict_list.append(nodule_dict) # 清空nodule_dict_list后，加入新uid的结节

                seriesuid_temp = seriesuid # 准备处理下一轮不同uid的结节数据

        count_image +=1
        print('\rplease wait... {:.2%}'.format((count_image - 100000) / 10000), end='', flush=True)
    
    # 末尾数据的处理
    nodule_uid_list.append(nodule_dict_list)

    # loop nodule_uid_list
    for each_nodule_dict_list in nodule_uid_list:

        # get list of slice
        list_slice = []
        for each_nodule_dict in each_nodule_dict_list:
            if not each_nodule_dict['slice'] in list_slice:
                list_slice.append(each_nodule_dict['slice'])

        # make .xml for each nodule
        for each_slice in list_slice:

            list_nodule = []
            list_x = []
            list_y = []
            list_w = []
            list_h = []
            list_count_image = []

            for each_nodule_dict in each_nodule_dict_list:
                if each_nodule_dict['slice'] == each_slice:
                    list_nodule.append(each_nodule_dict)

            for each_nodule in list_nodule:
                list_x.append(each_nodule['x'])
                list_y.append(each_nodule['y'])
                list_h.append(each_nodule['h'])
                list_w.append(each_nodule['w'])
                list_count_image.append(each_nodule['count_image'])

            # make .xml
            name_image = '{}.png'.format(min(list_count_image))
            tree = to_xml(
                name=name_image,
                list_x=list_x,
                list_y=list_y,
                list_w=list_w,
                list_h=list_h
            )

            # save .xml
            write_xml(tree, "data/LUNA16/negative/masked/Annotations/{:06d}.xml".format(min(list_count_image)))

def check_multi_nodule():
    pd_annotation = pd.read_csv(anno_path)
    count_image = 0

    seriesuid_temp = None
    slice_list = []
    multi_list = []
    for each_annotation in pd_annotation.iterrows():
        seriesuid = each_annotation[1].seriesuid
        coord_x = each_annotation[1].coordX
        coord_y = each_annotation[1].coordY
        coord_z = each_annotation[1].coordZ
        diameter_mm = each_annotation[1].diameter_mm

        mhd_name = '{}.mhd'.format(seriesuid)
        mhd_path = glob(os.path.join(working_path, '*', mhd_name), recursive=True)[0]

        numpyImage, numpyOrigin, numpySpacing = load_itk_image(mhd_path) # numpyImage.shape) 维度为(slice,w,h)

        # 将世界坐标下肺结节标注转换为真实坐标系下的坐标标注
        worldCoord = np.asarray([float(coord_x),float(coord_y),float(coord_z)])
        voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    
        slice = int(voxelCoord[2] + 0.5)

        if seriesuid_temp == seriesuid:
            if slice in slice_list:
                print('multi nodules in one image')
                multi_list.append((seriesuid_temp, slice))
            slice_list.append(slice)
        else:
            seriesuid_temp = seriesuid
            slice_list.clear()
            slice_list.append(slice)

        print(count_image)
        count_image += 1
    print(multi_list)

def get_main_txt():
    luna16_masked_path = 'data\LUNA16\masked'
    xml_path = os.path.join(os.path.join(luna16_masked_path, 'Annotations', '*.xml'))
    list_xml_path = glob(xml_path)

    # shuffle the list
    shuffle(list_xml_path)

    list_trainval = []
    list_test = []
    for xml_name in list_xml_path:
        file_name = int(os.path.basename(xml_name).split('.')[0])

        if len(list_trainval) <= int(1186 * 0.8):
            list_trainval.append(file_name)
        else:
            list_test.append(file_name)

    trainval_path = os.path.join(luna16_masked_path, 'ImageSets', 'Main','trainval.txt')
    f=open(trainval_path,'w')
    for i in list_trainval:
        f.write('{:06d}\n'.format(i))
    f.close()
    print('output: {}'.format(trainval_path))

    test_path = os.path.join(luna16_masked_path, 'ImageSets', 'Main', 'test.txt')
    f=open(test_path,'w')
    for i in list_test:
        f.write('{:06d}\n'.format(i))
    f.close()
    print('output: {}'.format(test_path))

def negative_get_main_txt():

    list_trainval = list(range(100000, 110000))
    test_path = os.path.join('data', 'LUNA16', 'negative', 'masked', 'ImageSets', 'Main', 'negative_trainval.txt')
    f=open(test_path,'w')
    for i in list_trainval:
        f.write('{:06d}\n'.format(i))
    f.close()
    print(
        'output: {}\n'
        'copy and paste the context to:'
        'data/LUNA16/masked/ImageSets/Main/trainval.txt'.format(test_path))

def augmentation_movement(): # 移动原结节在CT图像中的位置
    def beatau(e,level=0):
        if len(e) > 0:
            e.text='\n'+'\t'*(level+1)
            for child in e:
                beatau(child,level+1)
            child.tail=child.tail[:-1]
        e.tail='\n' + '\t'*level

    def to_xml(name, x, y, w, h):
        root = Element('annotation')#根节点
        erow1 = Element('folder')#节点1
        erow1.text= "VOC"
        
        
        erow2 = Element('filename')#节点2
        erow2.text= str(name)
        
        erow3 = Element('size')#节点3
        erow31 = Element('width')
        erow31.text = "512"
        erow32 = Element('height')
        erow32.text = "512"
        erow33 = Element('depth')
        erow33.text = "3" 
        erow3.append(erow31)
        erow3.append(erow32)
        erow3.append(erow33)

        root.append(erow1)
        root.append(erow2)
        root.append(erow3)

        erow4 = Element('object')
        
        erow41 = Element('name')
        erow41.text = 'nonnodule'

        erow4_pos = Element('pose')
        erow4_pos.text = 'Unspecified'

        erow4_tru = Element('truncated')
        erow4_tru.text = '0'

        erow4_dif = Element('difficult')
        erow4_dif.text = '0'

        erow42 = Element('bndbox')

        erow4.append(erow41)
        erow4.append(erow4_pos)
        erow4.append(erow4_tru)
        erow4.append(erow4_dif)
        erow4.append(erow42)

        erow421 = Element('xmin')
        erow421.text = str(x - np.round(w/2).astype(int))

        erow422 = Element('ymin')
        erow422.text = str(y - np.round(h/2).astype(int))

        erow423 = Element('xmax')
        erow423.text = str(x + np.round(w/2).astype(int))

        erow424 = Element('ymax')
        erow424.text = str(y + np.round(h/2).astype(int))

        erow42.append(erow421)
        erow42.append(erow422)
        erow42.append(erow423)
        erow42.append(erow424)

        root.append(erow4)

        beatau(root)      

        return ElementTree(root)

    def write_xml(tree, out_path):  
        '''''将xml文件写出 
        tree: xml树 
        out_path: 写出路径'''  
        tree.write(out_path, encoding="utf-8",xml_declaration=True)

    dir_anno_auto = 'data\\LUNA16\\masked\\Annotations_auto'
    # path_xml = os.path.join(dir_anno_auto, '*.xml')
    # list_xml_path =glob(path_xml)

    # make list_xml_path by trainval.txt
    path_trainval_txt = 'data\\LUNA16\\masked\\ImageSets\\Main\\trainval.txt'
    list_xml_path = []

    with open(path_trainval_txt, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            list_xml_path.append(os.path.join(dir_anno_auto, '{row}.xml'.format(row=row[0])))

    count_all = len(list_xml_path)
    count_xml = 0
    # input
    dir_image = 'data\\LUNA16\\masked\\JPEGImages'

    # output
    output_image = 'data\\LUNA16\\masked\\JPEGImages_move'
    output_annotation = 'data\\LUNA16\\masked\\Annotations_move'

    path_trainval_move = 'data\\LUNA16\\masked\\ImageSets\\Main\\trainval_move.txt'
    f = open(path_trainval_move, 'w', newline='')
    csv_writer = csv.writer(f)

    print('output: {}'.format(path_trainval_move))
    print('output: {}'.format(output_image))
    print('output: {}'.format(output_annotation))

    for i, each_xml_path in enumerate(list_xml_path):
        tree = ET.parse(each_xml_path)
        root = tree.getroot()

        # get image path
        file_name = '{:06d}.png'.format(int(root.find('filename').text.split('.')[0]))
        file_path = os.path.join(dir_image, file_name)

        list_object = root.findall('object')
        for j, each_object in enumerate(list_object):

            # get tht loc info
            bndbox = each_object.find('bndbox')
            x_min = int(bndbox.find('xmin').text)
            y_min = int(bndbox.find('ymin').text)
            x_max = int(bndbox.find('xmax').text)
            y_max = int(bndbox.find('ymax').text)

            for time in range(args.times_movement):

                # get the image
                image = cv2.imread(file_path)
                # cv2.imwrite('test\\test_original.png', image)

                # get crop image
                cropped = copy.deepcopy(image[y_min: y_max, x_min: x_max])
                # cv2.imwrite('test\\test_cropped.png', cropped)

                back_color = int(image[0, 0, 0])
                d = cropped.shape[0]

                # hide the original nodule
                image[y_min: y_max, x_min: x_max] = back_color

                # draw nodule
                if args.draw_nodule:
                    point_left_up =  (x_min, y_min)
                    point_right_down = (x_max, y_max)
                    cv2.rectangle(image, point_left_up, point_right_down, (0, 0, 255), 1)
                    cv2.imwrite('test\\test.png', image)

                count_loop = 0
                while True:
                    break_while = True

                    # get the position of moving the nodule to
                    x_random = random.randint(0, 512 - d)
                    y_random = random.randint(0, 512 - d)
                    area_nodule_move = image[y_random: y_random + d, x_random: x_random + d]

                    for y in range(d):
                        for x in range(d):
                            if int(area_nodule_move[y, x, 0]) == back_color:
                                break_while = False

                    count_loop += 1
                    if break_while or count_loop == 1000:
                        break

                # paste the nodule
                if not count_loop == 1000:
                    image[y_random: y_random + d, x_random: x_random + d] = cropped
                    output_file_name = '{:06d}_{j}_{time}.png'.format(
                        int(root.find('filename').text.split('.')[0]) + 200000,
                        j=j,
                        time=time,)
                    path_image_move = os.path.join(output_image, output_file_name)
                    cv2.imwrite(path_image_move, image)

                    # write .xml
                    tree = to_xml(
                        name=output_file_name,
                        x=int((x_min + x_max) / 2 + 0.5),
                        y=int((y_min + y_max) / 2 + 0.5),
                        w=32,
                        h=32)
                    write_xml(
                        tree, "data/LUNA16/masked/Annotations_move/{:06d}.xml".
                        format(200000 + count_xml))

                    # write csv
                    csv_writer.writerow([str(200000 + count_xml)], )
                count_xml += 1

        print('\rplease wait... {:.2%}'.format((i + 1) / count_all), end='', flush=True)

    f.close()

if __name__ == '__main__':
    if args.mode == 'get_masked_image':
        get_masked_image()
    elif args.mode == 'negative_get_masked_image':
        negative_get_masked_image()
    elif args.mode == 'negative_masked_image_rename':
        negative_masked_image_rename()

    elif args.mode == 'get_voc_anno':
        get_voc_anno()
    elif args.mode == 'check_multi_nodule':
        check_multi_nodule()
    elif args.mode == 'get_main_txt':
        get_main_txt()
    elif args.mode == 'negative_get_voc_anno':
        negative_get_voc_anno()
    elif args.mode == 'negative_get_main_txt':
        negative_get_main_txt()
    elif args.mode == 'augmentation_movement':
        augmentation_movement()
