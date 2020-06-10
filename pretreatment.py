import argparse
import csv
import glob
import os
from glob import glob
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
parser.add_argument('--mode', choices=['get_masked_image', 'get_voc_info', 'check_multi_nodule'], help='set the run mode')
args = parser.parse_args()

# set path
working_path = 'G:\lung_image\\all_LUNA16\\LUNA16'
cand_path = 'data/LUNA16/candidates.csv'
anno_path = 'data/LUNA16/annotations.csv'

# set debug mode
if args.debug:
    working_path = 'data/LUNA16/sample'

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
        path_img = os.path.join('data/LUNA16/masked/{}.png'.format(count_image))
        fig.savefig(path_img)
        plt.close()
        print(count_image)

        count_image += 1

def get_voc_info():
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
            erow42 = Element('bndbox')
            erow4.append(erow41)
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

        nodule_dict = {
            'slice': slice,
            'x': int(voxelCoord[0] + 0.5),
            'y': int(voxelCoord[1] + 0.5),
            'w': 32,
            'h': 32,
            'count_image': count_image,
        }

        # 针对第一个元素的处理
        if seriesuid_temp == None:
            nodule_dict_list.append(nodule_dict)
            seriesuid_temp = seriesuid

        else:
            # nodule_dict_list 增添元素 nodule_dict
            if seriesuid_temp == seriesuid:
                nodule_dict_list.append(nodule_dict)

            # nodule_uid_list 增添元素 nodule_dict_list
            else:                
                nodule_uid_list.append(nodule_dict_list)
                nodule_dict_list.clear()
                seriesuid_temp = seriesuid

        print(count_image)
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
            write_xml(tree, "data\LUNA16\masked\Annotations\{}.xml".format(min(list_count_image)))

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

if __name__ == '__main__':
    if args.mode == 'get_masked_image':
        get_masked_image()
    elif args.mode == 'get_voc_info':
        get_voc_info()
    elif args.mode == 'check_multi_nodule':
        check_multi_nodule()
    elif args.mode == 'get_voc_info':
        get_voc_info()
