import SimpleITK as sitk  # 医疗图片处理包
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob

path_working = 'G:\lung_image\\all_LUNA16\LUNA16'
cand_path = 'G:\lung_image\\all_LUNA16\luna16_backup\httpacademictorrentscom\CSVFILES\candidates.csv'
list_img_path= glob.glob(os.path.join(path_working, '*', '*.mhd'), recursive=True)

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

