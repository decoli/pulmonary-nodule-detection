import csv
import glob
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # 医疗图片处理包
from PIL import Image

from skimage import measure, morphology
from skimage.transform import resize
from sklearn.cluster import KMeans

path_working = 'G:\lung_image\\all_LUNA16\\LUNA16'
cand_path = 'G:\lung_image\\all_LUNA16\\luna16_backup\\httpacademictorrentscom\\CSVFILES\\candidates.csv'
anno_path = 'G:\lung_image\\all_LUNA16\\luna16_backup\\httpacademictorrentscom\\CSVFILES\\annotations.csv'
list_img_path= glob(os.path.join(path_working, '*', '*.mhd'), recursive=True)


for img_path in list_img_path:
    # img_path  = '/home/dataset/medical/luna16/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd'
    def load_itk_image(filename):
        itkimage = sitk.ReadImage(filename)
        numpyImage = sitk.GetArrayFromImage(itkimage)
        numpyOrigin = np.array(list(itkimage.GetOrigin()))  # CT原点坐标
        numpySpacing = np.array(list(itkimage.GetSpacing()))  # CT像素间隔
        return numpyImage, numpyOrigin, numpySpacing

    numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path)
    print(numpyImage.shape)  # 维度为(slice,w,h)
    print(numpyOrigin)
    print(numpySpacing)

    slice = 60
    image = np.squeeze(numpyImage[slice, ...])  # if the image is 3d, the slice is integer
    image_original = image
    plt.imshow(image,cmap='gray')
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
    mean = np.mean(image)
    std = np.std(image)
    image = image-mean
    image = image/std

    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,8))
    ax1.imshow(image_original,cmap='gray')
    plt.hist(image.flatten(),bins=200)
    plt.show()