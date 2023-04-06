# -*- coding: utf-8 -*-
"""
# @file name  : preprocess.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/3/22 14:31
# @brief      : 
"""
import math

import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('./imgs/test/75.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    h, w = binary.shape
    hors_k = int(math.sqrt(w) * 1.2)
    vert_k = int(math.sqrt(h) * 1.2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
    hors = ~cv2.dilate(binary, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
    vert = ~cv2.dilate(binary, kernel, iterations=1)
    borders = cv2.bitwise_or(hors, vert)

    cv2.imwrite('result.jpg', borders)


