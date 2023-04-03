# -*- coding: utf-8 -*-
"""
# @file name  : preprocess.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/3/22 14:31
# @brief      : 
"""
import cv2


from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv2.imread('./imgs/test/75.png')

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 如果灰度图像中的像素值大于165，则将像素值设为255，否则将像素值设为0
    # 而maxval参数表示最大像素值，也就是当像素值大于阈值时，设定的最大像素值。
    # 使用的最大像素值是255，表示将像素值大于阈值的像素设为白色（即最大像素值)。
    ret, binary = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)

    plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
    plt.show()
