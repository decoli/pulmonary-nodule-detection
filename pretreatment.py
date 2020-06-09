import argparse
import csv
import glob
import os
from glob import glob

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
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

path_working = 'G:\lung_image\\all_LUNA16\\LUNA16'
cand_path = 'G:\lung_image\\all_LUNA16\\luna16_backup\\httpacademictorrentscom\\CSVFILES\\candidates.csv'
anno_path = 'G:\lung_image\\all_LUNA16\\luna16_backup\\httpacademictorrentscom\\CSVFILES\\annotations.csv'

if args.debug:
    path_working = 'data/LUNA16/sample'
    cand_path = 'data/LUNA16/candidates.csv'
    anno_path = 'data/LUNA16/annotations.csv'

pd_annotation = pd.read_csv(anno_path)
print(pd_annotation)

for each_annotation in pd_annotation.iterrows():
    seriesuid = each_annotation[1].seriesuid
    coord_x = each_annotation[1].coordX
    coord_y = each_annotation[1].coordY
    coord_z = each_annotation[1].coordZ
    diameter_mm = each_annotation[1].diameter_mm

    name_mhd = '{}.mhd'.format(seriesuid)
    path_mhd = glob(os.path.join(path_working, '**', name_mhd), recursive=True)[0]

    def load_itk_image(filename):
        itkimage = sitk.ReadImage(filename)
        numpyImage = sitk.GetArrayFromImage(itkimage)
        numpyOrigin = np.array(list(itkimage.GetOrigin()))  # CT原点坐标
        numpySpacing = np.array(list(itkimage.GetSpacing()))  # CT像素间隔
        return numpyImage, numpyOrigin, numpySpacing

    numpyImage, numpyOrigin, numpySpacing = load_itk_image(path_mhd)
    print(numpyImage.shape)  # 维度为(slice,w,h)
    print(numpyOrigin)
    print(numpySpacing)

    slice = 60
    img = np.squeeze(numpyImage[slice, ...])  # if the img is 3d, the slice is integer
    image_original = img
    plt.imshow(img,cmap='gray')
    plt.show()

    '''read csv as a list of lists'''
    def readCSV(filename):
        lines = []
        with open(filename, "rt") as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                lines.append(line)
        return lines
    '''convert world coordinate to real coordinate'''
    def worldToVoxelCoord(worldCoord, origin, spacing):
        stretchedVoxelCoord = np.absolute(worldCoord - origin)
        voxelCoord = stretchedVoxelCoord / spacing
        return voxelCoord

    # 加载结节标注
    annos = readCSV(anno_path)  # 共1186个结节标注
    print(len(annos))
    print(annos[0:3])
    # 获取一个结节标注
    cand = annos[24]  
    print(cand)
    # 将世界坐标下肺结节标注转换为真实坐标系下的坐标标注
    worldCoord = np.asarray([float(cand[1]),float(cand[2]),float(cand[3])])
    voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    print(voxelCoord)

    # 由于肺部与周围组织颜色对比明显，考虑通过聚类的方法找到可区分肺区域和非肺区域的阈值，实现二值化。
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std

    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,8))
    ax1.imshow(image_original,cmap='gray')
    plt.hist(img.flatten(),bins=200)
    plt.show()

    # Kmean
    #提取肺部大致均值
    middle = img[100:400,100:400]  
    mean = np.mean(middle)  

    # 将图片最大值和最小值替换为肺部大致均值
    max = np.max(img)
    min = np.min(img)
    print(mean,min,max)
    img[img==max]=mean
    img[img==min]=mean


    image_array = img
    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,8))
    ax1.imshow(image_array,cmap='gray')
    ax2.hist(img.flatten(),bins=200)
    plt.show()

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)  
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    print('kmean centers:',centers)
    print('threshold:',threshold)
    '''
    kmean centers: [-0.2307924288649088, 1.472218336483015]
    threshold: 0.6207129538090531
    '''
    # 聚类完成后，清晰可见偏黑色区域为一类，偏灰色区域为另一类。
    image_array = thresh_img
    plt.imshow(image_array,cmap='gray') 
    plt.show()

    eroded = morphology.erosion(thresh_img,np.ones([4,4]))  
    dilation = morphology.dilation(eroded,np.ones([10,10]))  
    labels = measure.label(dilation)   
    fig,ax = plt.subplots(2,2,figsize=[8,8])
    ax[0,0].imshow(thresh_img,cmap='gray')  
    ax[0,1].imshow(eroded,cmap='gray') 
    ax[1,0].imshow(dilation,cmap='gray')  
    ax[1,1].imshow(labels)  # 标注mask区域切片图
    plt.show()

    label_vals = np.unique(labels)
    regions = measure.regionprops(labels) # 获取连通区域

    # 设置经验值，获取肺部标签
    good_labels = []
    for prop in regions:
        B = prop.bbox
        print(B)
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    '''
    (0L, 0L, 512L, 512L)
    (190L, 253L, 409L, 384L)
    (200L, 110L, 404L, 235L)
    '''
    # 根据肺部标签获取肺部mask，并再次进行’膨胀‘操作，以填满并扩张肺部区域
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    #imgs_to_process[i] = mask
    fig,ax = plt.subplots(2,2,figsize=[10,10])
    ax[0,0].imshow(img)  # CT切片图
    ax[0,1].imshow(img,cmap='gray')  # CT切片灰度图
    ax[1,0].imshow(mask,cmap='gray')  # 标注mask，标注区域为1，其他为0
    ax[1,1].imshow(img*mask,cmap='gray')  # 标注mask区域切片图
    plt.show()
