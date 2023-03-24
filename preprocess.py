# -*- coding: utf-8 -*-
"""
# @file name  : preprocess.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/3/22 14:31
# @brief      : 
"""
import os.path
import cv2
from glob import glob


if __name__ == '__main__':
    border_size = 50

    img_path = './imgs/test'
    img_files = glob(f'{img_path}/*')

    output_path = './imgs/processed_test'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in img_files:
        print(file)
        img = cv2.imread(file)
        height, width = img.shape[:2]
        print(f'original img size is width:{width}, height: {height}')

        # 定义边缘颜色（白色）
        border_color = [255, 255, 255]

        # 使用cv2.copyMakeBorder函数添加白色边缘
        img_with_border = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                             cv2.BORDER_CONSTANT, value=border_color)

        # 将图像转换为灰度图像
        gray = cv2.cvtColor(img_with_border, cv2.COLOR_BGR2GRAY)

        # 进行二值化处理
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # 将灰度图像二值化处理
        threshold_value = 200  # 阈值，可调整
        max_value = 255  # 最大像素值
        _, binary = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)

        # 进行形态学操作，去除小的噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 将表格设置为黑色，背景设置为白色
        result = cv2.bitwise_not(morph)

        file_name = os.path.basename(file)
        img_save_path = os.path.join(output_path, file_name)
        cv2.imwrite(img_save_path, result)
