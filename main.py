# -*- coding: utf-8 -*-
"""
# @file name  : main.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/3/22 13:20
# @brief      :
"""

import os
import cv2
from paddleocr import PPStructure,save_structure_res

if __name__ == "__main__":
    table_engine = PPStructure()

    save_folder = './output'
    img_path = './imgs/single.png'
    img = cv2.imread(img_path)
    result = table_engine(img)
    save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

    for line in result:
        line.pop('img')
        print(line)

