
# -*- coding: utf-8 -*-
'''
OpenCV中的霍夫变换
1.cv2.HoughLines(),返回一个(长度，角度)的数组，前者以像素为单位进行测量，后者以弧度为单位进行测量
2.该函数的第一个参数是二值图像，应用阈值或使用canny边缘检测，然后才应用霍夫变换。
  第二个参数和第三个参数分别为长度和角度，第四个参数是阈值。

'''
import math

import cv2
import numpy as np

img_path = './imgs/test/2.jpg'

img = cv2.imread(img_path)
h, w = img.shape[:2]       # 这里获得图像的长和高，为了下文获得图像的中心点

# 灰度变换，获得单通道图像，以便后文满足HoughLines的条件
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png', gray)

# 边缘检测，突出目标特征，使检测更加精准
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imwrite('edges.png', edges)

# 在进行霍夫变换之前要先将源图像进行二值化，或者进行 Canny 边缘检测
# lines=cv2.HoughLines（image,rho,theta,threshold）
#   image 是输入图像，即源图像，必须是 8 位的单通道二值图像。如果是其他类型的图像，在进行霍夫变换之前，需要将其修改为指定格式
#   rho 为以像素为单位的距离 r 的精度。一般情况下，使用的精度是 1。
#   theta 为角度 θ 的精度。一般情况下，使用的精度是 π/180，表示要搜索所有可能的角度。
#   threshold 是阈值. HoughLines 对直线所穿过的点的数量进行评估;
#                如果直线所穿过的点的数量小于阈值，则认为这些点恰好（偶然）在算法上构成直线，但是在源图像中该直线并不存在
#                如果阈值较小，就会得到较多的直线；阈值较大，就会得到较少的直线
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
print('lines', lines, 'len', len(lines))
print(lines[0])


# 获取霍夫线数组长度
tanh_value = None
for i in range(len(lines)):
    for rho, theta in lines[i]:
        # 注意： 这里的theta是垂直线的角度
        #        极坐标中的角度是按照顺时针计算的

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho   # 垂直线与直线的交点
        # 对直线的上下两端进行拓展，从而获得一定长度的直线
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # 在图上添加直线
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        angle_in_degree = theta * 180 / np.pi
        print("angle_in_degree: ", angle_in_degree)

        if i == 0:
            if x1 == x2 or y1 == y2:
                continue
            t = float(y2 - y1) / (x2 - x1)
            tanh_value = t

cv2.imwrite('img_w_houghlines.jpg', img)

rotate_angle = math.degrees(math.atan(tanh_value))
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
cv2.imwrite('rotated.jpg', rotated)

