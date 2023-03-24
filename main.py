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
    table_engine = PPStructure(
        show_log=True,
        det_model_dir='./models/ch_ppocr_server_v2.0_det_infer',
        # rec_model_dir='',
        # table_model_dir='',
        # layout_model_dir='',
    )

    save_folder = './output'
    img_path = './imgs/test/87.png'
    img = cv2.imread(img_path)
    result = table_engine(img)
    save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

    for line in result:
        line.pop('img')
        print(line)

