# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/3/15 10:26
# @brief      : 
"""
import os

import torch
import cv2
import numpy as np
import pandas as pd

# 加载模型
# 从网上找到训练好的模型.pth
model_path = os.path.join("")
model = torch.load(model_path, map_location=torch.device('cpu'))

# 加载待处理的表格图片
img_path = os.path.join("")
image = cv2.imread(img_path)

# 对图片进行预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.medianBlur(binary, 5)

# 将处理后的图片转换为tensor格式，并输入到模型中进行识别
tensor_img = torch.from_numpy(binary / 255).unsqueeze(0).unsqueeze(0).float()
pred = model(tensor_img)
pred = pred.squeeze(0).detach().numpy()

# 将预测结果转换为数据框格式
df = pd.DataFrame(pred, columns=['col_{}'.format(i+1) for i in range(pred.shape[1])])

# 将数据框输出为Excel文件
df.to_excel('table_output.xlsx', index=False)