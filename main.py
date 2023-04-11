# -*- coding: utf-8 -*-
"""
# @file name  : main.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/3/22 13:20
# @brief      :
"""

import os
import cv2
from paddleocr import PPStructure, save_structure_res
from ocr import ocr_img

if __name__ == "__main__":
    table_engine = PPStructure(
        layout=False,
        image_orientation=True,
        show_log=True,
        det_db_score_mode='slow',
        label_list=['0', '90', '180', '270'],
        # det_model_dir='./models/ch_ppocr_server_v2.0_det_infer',
        # rec_model_dir='./models/ch_ppocr_server_v2.0_rec_infer',
    )

    save_folder = './output'
    img_path = './imgs/test/14.png'
    img = cv2.imread(img_path)

    result = table_engine(img, return_ocr_result_in_table=True)
    orc_result = ocr_img(img_path)
    # layout_img_path = './result.jpg'
    # layout_result = table_engine(layout_img_path, return_ocr_result_in_table=False)

    boxes = []
    rec_res = []

    for line in orc_result[0]:
        coords = line[0]
        boxes.append(coords[0] + coords[2])
        rec_res.append(line[1])

    result[0]['res']['rec_res'] = rec_res
    result[0]['res']['boxes'] = boxes
    # result[0]['res']['cell_bbox'] = layout_result[0]['res']['cell_bbox']
    # result[0]['res']['html'] = layout_result[0]['res']['html']

    save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

